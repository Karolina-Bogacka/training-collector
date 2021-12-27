from typing import Dict, Callable
import socket

from flwr.server.strategy import Strategy
import flwr as fl

from config import SERVER_ADDRESS, FEDERATED_PORT
from pydloc.models import TCTrainingConfiguration
from src.strategy_manager import TCFedAvg


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def construct_strategy(id: int, data: TCTrainingConfiguration) -> Strategy:
    config_fn = get_on_fit_config_fn() if data.strategy == "custom" else None
    if data.strategy == "avg":
        return TCFedAvg(
            min_fit_clients=data.min_fit_clients,  # Minimum number of clients to be sampled for the next round
            min_available_clients=data.min_available_clients,
            min_eval_clients=data.min_available_clients,
            on_fit_config_fn=config_fn,
            id = id)
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
        # or from local memory, if we desire to do so
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
    strategy = construct_strategy(id, data)
    while is_port_in_use(FEDERATED_PORT+int(id)):
        id += 1
    fl.server.start_server(config={"num_rounds": data.num_rounds}, server_address=f"{SERVER_ADDRESS}:{FEDERATED_PORT + int(id)}",
                           strategy=strategy)
