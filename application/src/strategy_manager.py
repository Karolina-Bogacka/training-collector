import traceback
from logging import INFO
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
from flwr.common import Weights, Scalar, Parameters, EvaluateRes, parameters_to_weights
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from pydloc.models import Status, StatusEnum
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, InputLayer
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
#from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
            fraction_fit: float = 0.1,
            fraction_eval: float = 0.1,
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


class TCIFCA(fl.server.strategy.FedAvg):

    def __init__(
            self,
            num_rounds: int,
            fraction_fit: float = 0.1,
            fraction_eval: float = 0.1,
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
        jobs[self.id] = Status(status=StatusEnum.WAITING)

    pass


def get_cnn_model_1(input_shape):
    '''
    base_model = VGG16(input_shape=(224, 224, 3),  # Shape of our images
                       include_top=False,  # Leave out the last fully connected layer
                       weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)

    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = Dense(512, activation='relu')(x)

    # Add a dropout rate of 0.5
    x = Dropout(0.5)(x)

    # Add a final sigmoid layer with 1 node for classification output
    x = Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(base_model.input, x)
    '''
    model = tf.keras.Sequential()
    model.add(InputLayer(input_shape=(128, 128, 3)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    model.add(Dropout(0.3))

    # Adding flatten
    model.add(Flatten())

    # Adding full connected layer (dense)
    model.add(Dense(units=512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Adding output layer
    model.add(Dense(units=1, activation='sigmoid'))
    return model
