# FL Training Collector


Run python main.py to run the server.

Use:
python -m grpc_tools.protoc --proto_path=. protobufs/communications.proto --python_out=. --grpc_python_out=.
to compile protobuf output.

Use FastAPI functionalities to test the API on http://127.0.0.1:8000/docs.
Sample request body for post /job/config/{id}:
{
  "model": "cifar10",
  "epochs": 1,
  "rounds":1,
  "optimizer": "adam",
  "strategy": "fedavg",
  "batch_size": 32,
  "steps_per_epoch": 1
}