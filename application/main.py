import uvicorn
from fastapi import BackgroundTasks
from fastapi import FastAPI

from config import HOST, PORT
from pydloc.models import TCTrainingConfiguration, Status
from src import training_server
from src.strategy_manager import jobs

app = FastAPI()


# Receive training configuration
@app.post("/job/config/{id}/")
async def receive_conf(id, data: TCTrainingConfiguration, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(training_server.start_flower_server, id=id, data=data)
    except Exception as e:
        print("An exception occurred ::", e)
        return 500
    return 200


# Receive job status updates
@app.get("/job/status/{id}")
async def retrieve_status(id):
    if id in jobs:
        return jobs[id]
    else:
        return Status()


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
