import keras
import argparse
from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

parser = argparse.ArgumentParser(description='Argument parser.')
parser.add_argument('--model', '-m', help='Model path.')
parser.add_argument('--data', '-d', default=None)
args = parser.parse_args()

model = load_model(args.model)

if(False):
    data = args.data
    if(args.data == None):
        data = np.array([-0.33826175242575, 1.11959337641566, 1.04436655157316, -0.222187276738296, 0.49936080649727, -0.24676110061991, 0.651583206489972, 0.0695385865186387, -0.736727316364109, -0.366845639206541, 1.01761446783262, 0.836389570307029, 1.00684351373408, -0.443522816876142,
                         0.150219101422635, 0.739452777052119, -0.540979921943059, 0.47667726004282, 0.451772964394125, 0.203711454727929, -0.246913936910008, -0.633752642406113, -0.12079408408185, -0.385049925313426, -0.0697330460416923, 0.0941988339514961, 0.246219304619926, 0.0830756493473326, 3.68],
                        )
        data = np.expand_dims(data, axis=0)
    print(data.shape)
    res = model.predict(data, verbose=0)
    print(res)

#! Predict single image
if(True):
    x = load_img(args.data, target_size=(299, 299))
    x = img_to_array(x)
    x = x/255.
    x = np.expand_dims(x, axis=0)
    res = model.predict(x)
    print(res)
