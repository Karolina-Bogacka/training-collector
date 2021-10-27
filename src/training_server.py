from flwr.server.strategy import Strategy
import flwr as fl


def start_flower_server(num_rounds: int, strategy: str):
    if strategy == "fedavg":
        strategy_practical = fl.server.strategy.FedAvg(  # Sample 10% of available clients for the next round
            min_fit_clients=1,  # Minimum number of clients to be sampled for the next round
            min_available_clients=1,
            # Minimum number of clients that need to be connected to the server before a training round can start
        )
        fl.server.start_server(config={"num_rounds": num_rounds}, strategy=strategy_practical)
