import uvicorn
from fastapi import FastAPI

from pydloc.models import TrainingConfiguration
from src.training_server import start_flower_server
from fastapi import BackgroundTasks

app = FastAPI()


# Receive weight updates
@app.put("/model/update/{id}/{version}")
async def receive_updated(id, version):
    # A request to the Orchestrator?
    return "Weights Received"


# Receive training configuration
@app.post("/job/config/{id}/")
async def receive_conf(id, data: TrainingConfiguration, background_tasks: BackgroundTasks):
    background_tasks.add_task(start_flower_server, num_rounds=data.rounds, strategy=data.strategy)
    return "Received"


# Receive job status updates
@app.get("/job/status/{id}")
async def retrieve_status(id):
    return "Going Great"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
