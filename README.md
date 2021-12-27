# FL Training Collector


Run python main.py to run the server.

Use:
python -m grpc_tools.protoc --proto_path=. protobufs/communications.proto --python_out=. --grpc_python_out=.
to compile protobuf output.

Use FastAPI functionalities to test the API on http://127.0.0.1:8000/docs.
Sample request body for post /job/config/{id}:
{
"strategy" : "custom",
"model_id" : "base",
"num_rounds" : "3",
"min_fit_clients" : "4",
"min_available_clients": "6",
"adapt_config": "custom",
"config":[{
      "config_id" : "min_effort",
      "batch_size": "64",
      "steps_per_epoch" : "32",
      "epochs" : "500",
      "learning_rate" : "0.001"
      }]
}