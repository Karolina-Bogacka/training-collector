import socket
from typing import Dict, Callable

import flwr as fl
import tensorflow as tf
from config import FEDERATED_PORT
from flwr.server.strategy import Strategy
from pydloc.models import TCTrainingConfiguration
from src.strategy_manager import TCCifarFedAvg



def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def construct_strategy(id: int, data: TCTrainingConfiguration, model=None) -> Strategy:
    config_fn = get_on_fit_config_fn() if data.strategy == "custom" else None
    if data.strategy == "avg":
        return TCCifarFedAvg(
            num_rounds=data.num_rounds,
            min_fit_clients=data.min_fit_clients,  # Minimum number of clients to be sampled for the next round
            min_available_clients=data.min_available_clients,
            min_eval_clients=data.min_fit_clients,
            blacklisted=int(data.blacklisted),
            id=id)
    elif data.strategy == "fast-and-slow":
        return fl.server.strategy.FastAndSlow(
            min_fit_clients=data.min_fit_clients,  # Minimum number of clients to be sampled for the next round
            min_available_clients=data.min_available_clients,
            min_eval_clients=data.min_available_clients,
            on_fit_config_fn=config_fn)
    elif data.strategy == "fault-tolerant":
        return fl.server.strategy.FaultTolerantFedAvg(
            min_fit_clients=data.min_fit_clients,  # Minimum number of clients to be sampled for the next round
            min_available_clients=data.min_available_clients,
            min_eval_clients=data.min_available_clients,
            on_fit_config_fn=config_fn
        )
    elif data.strategy == "fed-adagrad":
        return fl.server.strategy.FedAdagrad(
            min_fit_clients=data.min_fit_clients,  # Minimum number of clients to be sampled for the next round
            min_available_clients=data.min_available_clients,
            min_eval_clients=data.min_available_clients,
            on_fit_config_fn=config_fn
        )
    elif data.strategy == "fedadam":
        return fl.server.strategy.FedAdam(
            min_fit_clients=data.min_fit_clients,  # Minimum number of clients to be sampled for the next round
            min_available_clients=data.min_available_clients,
            min_eval_clients=data.min_available_clients,
            on_fit_config_fn=config_fn
        )
    elif data.strategy == "fedyogi":
        return fl.server.strategy.FedYogi(
            min_fit_clients=data.min_fit_clients,  # Minimum number of clients to be sampled for the next round
            min_available_clients=data.min_available_clients,
            min_eval_clients=data.min_available_clients,
            on_fit_config_fn=config_fn
        )
    elif data.strategy == "q-avg":
        return fl.server.strategy.QFedAvg(
            min_fit_clients=data.min_fit_clients,  # Minimum number of clients to be sampled for the next round
            min_available_clients=data.min_available_clients,
            min_eval_clients=data.min_available_clients,
            on_fit_config_fn=config_fn
        )
    else:
        # TODO: perform a lookup to get an existing custom strategy from repository
        return fl.server.strategy.FedAvg(
            min_fit_clients=data.min_fit_clients,  # Minimum number of clients to be sampled for the next round
            min_available_clients=data.min_available_clients,
            min_eval_clients=data.min_available_clients,
            on_fit_config_fn=config_fn
        )


def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "learning_rate": str(0.001 * pow(0.1, rnd / 5)),
            "batch_size": str(32),
        }
        return config

    return fit_config


def start_flower_server(id: int, data: TCTrainingConfiguration):
    model = tf.keras.applications.EfficientNetB0(
            input_shape=(32, 32, 3), weights=None, classes=10
        )
    adam_opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    model.compile(adam_opt, "sparse_categorical_crossentropy", metrics=[
            "accuracy"])
    strategy = construct_strategy(id, data, model)
    fl.server.start_server(config={"num_rounds": data.num_rounds},
                           server_address=f"[::]:{FEDERATED_PORT}",
                           strategy=strategy)
