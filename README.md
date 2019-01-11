# ML-SANDBOX

## TF Docker for local development

<!-- Docker build -->

```bash
docker build --rm -f Dockerfile -t mytensorflow .
```

This configuration only works for my machine. If you want to use this you have to adjust configuration to fit your computer.

```bash
docker run --rm -it -p 6006:6006/tcp -p 8888:8888/tcp --mount type=bind,source="C:/Users/i849438/Development/git/repos/ml-sandbox/notebooks",target=/notebooks/custom --mount type=bind,source="C:/Users/i849438/Development/git/repos/ml-sandbox",target=/development --mount type=bind,source="C:/Users/i849438/OneDrive - SAP SE\Data",target=/data mytensorflow
```

## TF Serving Docker

Instructions from tensorflow website.

https://www.tensorflow.org/serving/docker

<!-- sudo docker run -p 8501:8501 --mount type=bind,source=/tf_serving/models/FruitModel,target=/models/FruitModel \
-e MODEL_NAME=FruitModel -t tensorflow/serving -->

```bash
sudo docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source=/tf_serving/models/FruitModel,target=/models/FruitModel --mount type=bind,source=/tf_serving/models/NumModel,target=/models/NumModel --mount type=bind,source=/tf_serving/models.config,target=/models/models.config -t tensorflow/serving --model_config_file=/models/models.config
```

Can test `NumModel` which expects a single number input and returns a single number output.

```bash
curl -d '{"instances": 1}' -X POST http://localhost:8501/models/NumModel:predict
```

## Image Retraining

### TF Scripts

https://www.tensorflow.org/hub/tutorials/image_retraining

Using the TF scripts retrain.py (using any of the models from tf hub https://tfhub.dev/) and the label_image.py.

```bash
curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py
```

```bash
curl -LO https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py
```

Example of usage...

Train

```bash
python retrain.py --image_dir="\Users\i849438\OneDrive - SAP SE\Data\ml_data\images\oilwater" --tfhub_module=https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2 --random_crop=10 --random_scale=10 --random_brightness=10 --flip_left_right --saved_model_dir=models\oilwater --how_many_training_steps=50
```

The training outputs to the /tmp/output_graph.pb, along with the /tmp/output_labels.txt. This is a DIFFERENT file than the saved_model_dir file. The /tmp/ file is required for the inference step next.

Inference

```bash
python label_image.py --graph=models\pb\oilwater_mobilenet\output_graph.pb --labels=models\pb\oilwater_mobilenet\output_labels.txt --input_height=224 --input_width=224 --input_layer=Placeholder --output_layer=final_result --image=images\oil.png
```

Here is a resource explaining different tf-hub module types...
https://medium.com/ymedialabs-innovation/how-to-use-tensorflow-hub-with-code-examples-9100edec29af
