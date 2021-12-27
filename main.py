import uvicorn
from fastapi import FastAPI

from config import HOST, SERVER_PORT
from pydloc.models import TCTrainingConfiguration
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
async def receive_conf(id, data: TCTrainingConfiguration, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(start_flower_server, id=id, data=data)
    except Exception as e:
        print("An exception occurred ::", e)
        return 500
    return 200


# Receive job status updates
@app.get("/job/status/{id}")
async def retrieve_status(id):
    return "Going Great"


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=SERVER_PORT)
