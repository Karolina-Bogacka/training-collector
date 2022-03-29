# FL Training Collector

Run `docker-compose -f docker-compose.yml up` to run the server.

Use FastAPI functionalities to test the API on http://127.0.0.1:8000/docs.
Sample request body for post /job/config/{id}:
{
"strategy" : "avg",
"model_id" : "base",
"num_rounds" : "3",
"min_fit_clients" : "1",
"min_available_clients": "1",
"adapt_config": "custom",
"config":[{
      "config_id" : "min_effort",
      "batch_size": "64",
      "steps_per_epoch" : "32",
      "epochs" : "5",
      "learning_rate" : "0.001"
      }]
}