# ML-SANDBOX

## TF Docker for local development

<!-- Docker build -->

`docker build --rm -f Dockerfile -t mytensorflow .`

This configuration only works for my machine. If you want to use this you have to adjust configuration to fit your computer.

`docker run --rm -it -p 6006:6006/tcp -p 8888:8888/tcp --mount type=bind,source=C:/Users/i849438/Development/git/repos/ml-sandbox/notebooks,target=/notebooks/custom --mount type=bind,source=C:/Users/i849438/Development/git/repos/ml-sandbox,target=/development --mount type=bind,source="C:/Users/i849438/OneDrive - SAP SE\Data",target=/data mytensorflow`

## TF Serving Docker

Instructions from tensorflow website.

https://www.tensorflow.org/serving/docker

<!-- sudo docker run -p 8501:8501 --mount type=bind,source=/tf_serving/models/FruitModel,target=/models/FruitModel \
-e MODEL_NAME=FruitModel -t tensorflow/serving -->

sudo docker run -p 8500:8500 -p 8501:8501 \
 --mount type=bind,source=/tf_serving/models/FruitModel,target=/models/FruitModel \
 --mount type=bind,source=/tf_serving/models/NumModel,target=/models/NumModel \
 --mount type=bind,source=/tf_serving/models.config,target=/models/models.config \
 -t tensorflow/serving --model_config_file=/models/models.config

Can test `NumModel` which expects a single number input and returns a single number output.

`curl -d '{"instances": 1}' -X POST http://localhost:8501/models/NumModel:predict`
