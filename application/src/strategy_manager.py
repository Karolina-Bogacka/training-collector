import traceback
from logging import INFO
from typing import List, Tuple, Optional, Callable, Dict
import os
import pickle
import time
import traceback
from typing import List, Tuple, Optional, Callable, Dict

import flwr as fl
import pandas as pd
import requests
import tensorflow as tf
from PIL import Image
from config import REPOSITORY_ADDRESS, JSON_FILE
from flwr.common import Weights, Scalar, Parameters, EvaluateRes, parameters_to_weights
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from keras.utils.np_utils import to_categorical
from pydloc.models import Status, StatusEnum
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, InputLayer
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
import flwr as fl
import os
import gc
import pandas as pd
import numpy as np
from flwr.client import start_numpy_client
from flwr.common.logger import log
from keras import backend, Sequential
from keras.constraints import maxnorm
from keras.datasets.cifar import load_batch
from keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
from starlette.concurrency import run_in_threadpool
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
IMAGE_SIZE = (32, 32)
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


class TCCifarFedAvg(CustomFedAvg):

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
            initial_parameters = None,
            id: int = -1,
            blacklisted: int = 0
    ) -> None:
        super().__init__(fraction_fit, fraction_eval, min_fit_clients, min_eval_clients,
                         min_available_clients, eval_fn, on_fit_config_fn, on_evaluate_config_fn,
                         accept_failures, initial_parameters, blacklisted)
        self.id = id
        self.num_rounds = num_rounds
        self.eval_fn = eval_fn
        data = {"loss": [], "accuracy": []}
        self.results = pd.DataFrame(data)
        self.losses = []
        self.times = []
        self.evaluated = []
        log(INFO, f"{os.path.join(os.sep, 'german-traffic', 'Test.csv')}")
        self.y_test = pd.read_csv(os.path.join(os.sep, 'german-traffic', 'Test.csv'))
        labels = self.y_test["ClassId"].values
        imgs = self.y_test["Path"].values
        labels = np.array(labels)
        data2 = []

        # Retreiving the images

        for img in imgs:
            image = Image.open(os.path.join(os.sep, 'german-traffic', img))
            image = image.resize([32, 32])
            data2.append(np.array(image))

        self.x_test = np.array(data2)
        self.y_test = to_categorical(labels, 43)
        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu',
                              input_shape=self.x_test.shape[1:]))
        self.model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(43, activation='softmax'))

        # Compilation of the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy'])

        jobs[self.id] = Status(status=StatusEnum.WAITING)

    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        self.model.set_weights(parameters_to_weights(aggregated_weights[0]))
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
        with open(os.path.join(os.sep, "code", "application", "results.pkl"),
                  'wb') as handle:
            self.losses.append(results[0])
            self.times.append(time.time())
            losses, accuracy = self.model.evaluate(self.x_test, self.y_test, 32)
            self.evaluated.append((losses, accuracy))
            results_to_file = {"loss": self.losses,
                       "times": self.times, "evaluate":self.evaluated}
            pickle.dump(results_to_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.model.save("/code/application/model")
        log(INFO, results)
        return results

