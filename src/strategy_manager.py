import os
import pickle
import traceback
from typing import List, Tuple, Optional, Callable, Dict

import flwr as fl
import requests
from flwr.common import Weights, Scalar, Parameters, EvaluateRes
from flwr.server.client_proxy import ClientProxy

from config import REPOSITORY_ADDRESS, JSON_FILE
from pydloc.models import Status, StatusEnum

if os.path.isfile(os.path.join("..", JSON_FILE)):
    with open(os.path.join("..", JSON_FILE), 'rb') as handle:
        jobs = pickle.load(handle)
else:
    jobs = {}


class TCFedAvg(fl.server.strategy.FedAvg):

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
                        r = requests.post(f"{REPOSITORY_ADDRESS}/model/{self.id}/{rnd}", files={"file": f})
                    except requests.exceptions.RequestException as e:
                        print(f"Failed to send weights of job {self.id} to repository")
                        traceback.print_exc()
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
        return super().aggregate_evaluate(rnd, results, failures)
