import traceback
from logging import INFO, WARNING, DEBUG
from random import random
from typing import List, Tuple, Optional, Callable, Dict
import os
import pickle
import traceback
from typing import List, Tuple, Optional, Callable, Dict

import flwr as fl
import pandas as pd
import requests
import tensorflow as tf
from config import REPOSITORY_ADDRESS, JSON_FILE
from flwr.common import Weights, Scalar, Parameters, EvaluateRes, parameters_to_weights, \
    weights_to_parameters, FitIns, EvaluateIns
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW, FedAvg
from keras import Sequential
from keras.constraints import maxnorm
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
from pydloc.models import Status, StatusEnum
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, InputLayer
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
#from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from application.src.custom_strategy import CustomFedAvg

if os.path.isfile(os.path.join("..", JSON_FILE)):
    with open(os.path.join("..", JSON_FILE), 'rb') as handle:
        jobs = pickle.load(handle)
else:
    jobs = {}

ERAS = 50
EPOCHS = 1
SEED = 42
BATCH_SIZE = 16
IMAGE_SIZE = (128, 128)
PREFETCH_BUFFER_SIZE = 400
SHUFFLE_BUFFER_SIZE = 1000
CACHE_DIR = "caches/ds_cache"
ds_params = dict(
    labels="inferred",
    label_mode="categorical",
    class_names=["all", "hem"],
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    seed=SEED
)


def load_image(path):
    image = tf.io.decode_bmp(tf.io.read_file(path), channels=3)
    return image


def preprocess(image):
    result = tf.image.resize(image, (128, 128))
    result = tf.image.per_image_standardization(result)
    return result


def get_ds(filenames, labels, batch_size, pref_buf_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    label_ds, image_pathes = tf.data.Dataset.from_tensor_slices(labels), tf.data.Dataset.from_tensor_slices(filenames)
    images_ds = image_pathes.map(load_image, AUTOTUNE).map(preprocess, AUTOTUNE)
    ds = tf.data.Dataset.zip((images_ds, label_ds)).batch(batch_size).prefetch(pref_buf_size)
    return ds


def test_model(model, callbacks=None):
    test_dir = os.path.join(os.sep, 'data', "validation_data")
    test_data_csv = pd.read_csv(
        test_dir + "/C-NMC_test_prelim_phase_data_labels.csv"
    )
    test_data_dir = test_dir + "/C-NMC_test_prelim_phase_data_labels.csv"
    dir_list = list(os.walk(test_data_dir))[0]
    filenames = sorted([test_data_dir + "/" + name for name in dir_list[2]])
    get_label_by_name = lambda x: test_data_csv.loc[test_data_csv['new_names'] == x]["labels"].to_list()[0]
    labels = [1 - get_label_by_name(name) for name in dir_list[2]]
    # print(filenames)
    # print(test_data_csv[["new_names", "labels"]])
    test_ds = get_ds(filenames, labels, BATCH_SIZE, PREFETCH_BUFFER_SIZE)
    if callbacks == None:
        loss, accuracy, precision = model.evaluate(test_ds)
    else:
        loss, accuracy, precision = model.evaluate(test_ds, callbacks=callbacks)

    return loss, {'accuracy': accuracy, 'precision': precision}


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    test_dir = os.path.join(os.sep, 'data', "validation_data")
    test_data_csv = pd.read_csv(
        test_dir + "/C-NMC_test_prelim_phase_data_labels.csv"
    )
    test_data_dir = test_dir + "/C-NMC_test_prelim_phase_data"
    dir_list = list(os.walk(test_data_dir))[0]
    filenames = sorted([test_data_dir + "/" + name for name in dir_list[2]])
    get_label_by_name = lambda x: test_data_csv.loc[test_data_csv['new_names'] == x]["labels"].to_list()[0]
    labels = [str(1 - get_label_by_name(name)) for name in dir_list[2]]
    idg = ImageDataGenerator()
    df = pd.DataFrame({'images': filenames, 'labels': labels})

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model.set_weights(weights)  # Update model with the latest parameters
        test_gen = idg.flow_from_dataframe(dataframe=df,
                                           directory=test_dir,
                                           x_col='images',
                                           y_col='labels',
                                           class_mode='categorical',
                                           target_size=IMAGE_SIZE,
                                           color_mode='rgb',
                                           batch_size=BATCH_SIZE)
        loss, accuracy, precision = model.evaluate(test_gen, steps=len(filenames) // BATCH_SIZE)
        return (loss, {'accuracy': accuracy, 'precision': precision})

    return evaluate


class TCCifarFedAvg(fl.server.strategy.FedAvg):

    def __init__(
            self,
            num_rounds: int,
            fraction_fit: float = 1,
            fraction_eval: float = 1,
            min_fit_clients: int = 2,
            min_eval_clients: int = 2,
            min_available_clients: int = 2,
            eval_fn: Optional[
                Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            id: int = -1
    ) -> None:
        super().__init__(fraction_fit, fraction_eval, min_fit_clients, min_eval_clients,
                         min_available_clients, eval_fn, on_fit_config_fn, on_evaluate_config_fn,
                         accept_failures, initial_parameters)
        self.id = id
        self.num_rounds = num_rounds
        self.eval_fn = eval_fn
        data = {"loss":[], "accuracy":[]}
        self.results = pd.DataFrame(data)
        jobs[self.id] = Status(status=StatusEnum.WAITING)

    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        jobs[self.id] = Status(status=StatusEnum.TRAINING)
        if rnd == self.num_rounds:
            if aggregated_weights is not None:
                # Save aggregated_weights
                print(f"Saving final aggregated_weights...")
                path = f"model/{self.id}"
                is_exist = os.path.exists(path)
                if not is_exist:
                    # Create a new directory because it does not exist
                    os.makedirs(path)
                pickle.dump(aggregated_weights, open(f"{path}/aggregated-weights.sav", 'wb'))
                with open(f"{path}/aggregated-weights.sav", 'rb') as f:
                    try:
                        pass
                    except requests.exceptions.RequestException as e:
                        print(f"Failed to send weights of job {self.id} to repository")
                        traceback.print_exc()

                os.remove(f"{path}/aggregated-weights.sav")
                os.rmdir(f"{path}")
            jobs[self.id] = Status(status=StatusEnum.FINISHED, round=rnd)
        else:
            jobs[self.id] = Status(status=StatusEnum.TRAINING, round=rnd)
        return aggregated_weights

    def aggregate_evaluate(self,
                           rnd: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[BaseException],
                           ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        jobs[self.id] = Status(status=StatusEnum.FINISHED)
        with open(os.path.join("..", JSON_FILE), 'wb') as handle:
            pickle.dump(jobs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        results = super().aggregate_evaluate(rnd, results, failures)
        log(INFO, results)
        return results


class TCCifarIFCA(CustomFedAvg):

    def __init__(
            self,
            num_rounds: int,
            fraction_fit: float = 1,
            fraction_eval: float = 1,
            min_fit_clients: int = 12,
            min_eval_clients: int = 12,
            min_available_clients: int = 12,
            eval_fn: Optional[
                Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            id: int = -1,
            local_epochs: int = 16,
            cluster_number: int = 2,
            blacklisted:int=1,

    ) -> None:
        super().__init__(fraction_fit, fraction_eval, min_fit_clients, min_eval_clients,
                         min_available_clients, eval_fn, on_fit_config_fn,
                         on_evaluate_config_fn,
                         accept_failures, initial_parameters, blacklisted=blacklisted)
        self.id = id
        if (
                min_fit_clients > min_available_clients
                or min_eval_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.cluster_number = cluster_number
        self.clients = {}
        self.cid_maxes = {}
        (self.x_train, self.y_train), (self.x_test, self.y_test) = \
            tf.keras.datasets.cifar10.load_data()
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)
        jobs[self.id] = Status(status=StatusEnum.WAITING)
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu',
                              padding='same'))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='softmax'))
        lrate = 0.01
        decay = lrate / 50
        sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,
                           metrics=['accuracy'])
        self.models = [self.model.get_weights() for _ in
                       range(self.cluster_number)]
        self.cluster_results = [[[]*self.num_rounds] for _ in range(self.cluster_number)]
        self.current_epoch = 0
        self.blacklisted=blacklisted
        self.blacklist = []
        self.weights = {}

    def configure_fit(
            self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        to_send = []
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        if self.current_epoch < self.local_epochs:
            self.clients = {}
            if self.blacklisted > 0:
                if rnd == 0:
                    # form the blacklist by sampling n clients
                    cids = [client.cid[0:-5] for client in clients]
                    to_blacklist = random.sample(cids, self.blacklisted)
                    self.blacklist = [client for client in clients if client.cid[0:-5] in
                                      to_blacklist]
            log(INFO, f"Blacklisted {self.blacklist} from clients {clients}")
            for client in clients:
                common_cid = client.cid[0:-5]
                if common_cid not in self.clients:
                    self.clients[common_cid] = 0
                else:
                    self.clients[common_cid] += 1

                if self.on_fit_config_fn is not None:
                    # Custom fit config function provided
                    config = self.on_fit_config_fn(rnd)
                    config["model_index"] = self.clients[common_cid]
                    config["cluster_number"] = self.cluster_number
                    config["local_epochs"] = self.local_epochs
                else:
                    config = {"model_index": self.clients[common_cid], "cluster_number":
                        self.cluster_number,
                              "local_epochs": self.local_epochs}
                fit_ins = FitIns(weights_to_parameters(self.models[self.clients[
                    common_cid]]), config)
                if client in self.blacklist:
                    if rnd % 2 == 1:
                        fit_ins = self.weights[client]
                    else:
                        self.weights[client] = fit_ins
                to_send.append((client, fit_ins))
                log(INFO, f"Current cid client code is {client.cid}")
        else:
            self.clients = {}
            sent_cids = []
            for client in clients:
                common_cid = client.cid[0:-5]
                if common_cid not in sent_cids:
                    sent_cids.append(common_cid)
                    cluster_index = self.cid_maxes[common_cid]
                    if self.on_fit_config_fn is not None:
                        # Custom fit config function provided
                        config = self.on_fit_config_fn(rnd)
                        config["model_index"] = cluster_index
                        config["cluster_number"] = self.cluster_number
                        config["local_epochs"] = self.local_epochs
                    else:
                        config = {"model_index": cluster_index, "cluster_number":
                            self.cluster_number,"local_epochs": self.local_epochs}
                    fit_ins = FitIns(weights_to_parameters(self.models[self.cid_maxes[
                        common_cid]]), config)
                    to_send.append((client, fit_ins))
                    log(INFO, f"Current cid client code is {client.cid} with cluster "
                              f"number {self.cid_maxes[common_cid]}")
        # Return client/config pairs
        return to_send

    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        aggregated_params, metrics = super().aggregate_fit(rnd, results, failures)

        # Divide results between existing clusters
        cluster_results = [[] for _ in range(self.cluster_number)]
        if self.blacklisted > 0:
            log(DEBUG,"Performing blacklist")
            for result in results:
                if result[0] in self.blacklist:
                    log(DEBUG, f"{result[0]} in blacklist")
                    self.weights[result[0]][0] = parameters_to_weights(result[1].parameters)
            if rnd % 2 == 1:
                for result in results:
                    if result[0] in self.blacklist:
                        results.remove(result)
        for result in results:
            if result[1].metrics["clustering_phase"]:
                if not result[1].metrics["failure"]:
                    cluster_index = result[1].metrics['assigned_cluster']
                    cluster_results[cluster_index].append(result)
                    self.cid_maxes[result[0].cid[0:-5]] = cluster_index
                    log(INFO, f"Current cluster index for {result[0].cid} cid is"
                              f" {cluster_index} with min loss {result[1].metrics['loss']}")
                else:
                    log(INFO, f"The invalid cid client code is {result[0].cid}")
            else:
                cluster_index = self.cid_maxes[result[0].cid[0:-5]]
                cluster_results[cluster_index].append(result)
        # Update weights of clustered models
        for i in range(self.cluster_number):
            if cluster_results[i]:
                aggregated_params_local, metrics_local = super().aggregate_fit(rnd,
                                                                   cluster_results[i],
                                                               [])
                self.models[i] = parameters_to_weights(aggregated_params_local)
        self.current_epoch += 1
        return aggregated_params, metrics

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        to_send = []
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        self.clients = {}
        sent_cids = []
        for client in clients:
            common_cid = client.cid[0:-5]
            if common_cid not in sent_cids:
                sent_cids.append(common_cid)
                cluster_index = self.cid_maxes[common_cid]
                config = {"model_index": cluster_index, "cluster_number":
                        self.cluster_number, "local_epochs": self.local_epochs}
                eval_ins = EvaluateIns(weights_to_parameters(self.models[
                                                                self.cid_maxes[
                    common_cid]]), config)
                to_send.append((client, eval_ins))
                log(INFO, f"Current cid client code is {client.cid} with cluster "
                          f"number {self.cid_maxes[common_cid]}")
        # Return client/config pairs
        return to_send

    def aggregate_evaluate(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        log(INFO, f"aggregating {results}")
        cluster_results = [[] for _ in range(self.cluster_number)]
        for result in results:
            cluster_index = self.cid_maxes[result[0].cid[0:-5]]
            cluster_results[cluster_index].append(result)
        for i in range(self.cluster_number):
            if cluster_results[i]:
                self.model.set_weights(self.models[i])
                losses, accuracy = self.model.evaluate(self.x_test, self.y_test, 32)
                eval_results = super().aggregate_evaluate(rnd,cluster_results[i],[])
                eval_results = eval_results+(losses,accuracy)
                log(INFO, f"Current eval results {eval_results}")
                self.cluster_results[i][rnd] = eval_results
                self.model.save(os.path.join(os.sep, "code", "application", "model",
                                             f"cluster-{i}"))
        with open(os.path.join(os.sep, "code", "application", "results.pkl"),'wb') as handle:
            pickle.dump(self.cluster_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return super().aggregate_evaluate(rnd, results, failures)
