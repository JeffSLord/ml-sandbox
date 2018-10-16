# region     #! Imports
import argparse
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Conv1D, MaxPooling1D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.callbacks import TensorBoard
from keras import optimizers
import numpy as np
import os
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# endregion
file_path, file_name = os.path.split(__file__)
print(file_path)

# region     #! Arguments
parser = argparse.ArgumentParser(description='Argument parser.')
parser.add_argument('--dir', '-d', help='Path to data')
parser.add_argument('--epochs', '-e', default=10, type=int)
parser.add_argument('--batch_size', '-b', default=32, type=int)
parser.add_argument('--validation_split', '-v', default=0.2)
parser.add_argument('--save_path', '-s',
                    default=os.path.join(file_path, "k_dnn_model.h5"))
parser.add_argument('--log_dir', '-log',
                    default=os.path.join(file_path, "logs/"))
parser.add_argument('--load_model', '-l', default=None)
parser.add_argument('--predict', '-p', default=False)
# ? ...
args = parser.parse_args()
# endregion


# region    #! Process Data
print("[INFO] Processing data...")
data = pd.read_csv(args.dir, quotechar='"')
data = shuffle(data)
data = data.as_matrix()
# data = np.genfromtxt(args.dir, skip_header=1)
y_data = data[:, -1]
y_data = y_data.astype(int)
x_data = data[:, 1: -1]

print(y_data[0])
print(x_data[0])

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2)

one_hot_train = keras.utils.to_categorical(y_train, num_classes=2)
one_hot_test = keras.utils.to_categorical(y_test, num_classes=2)

entries, features = x_data.shape
print(x_data.shape)
print(y_data.shape)
# endregion

# region    #! Define Model
print("[INFO] Defining model...")
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=features))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# endregion

# region    #! Compile Model
print("[INFO] Compiling model...")
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# endregion


# region    #! Train Model
print("[INFO] Training model...")
if(args.load_model != None and os.path.isfile(args.load_model)):
    print("[INFO] Continuing training from loaded model...")
    model = load_model(args.load_model)
elif(os.path.isfile(args.save_path)):
    print("[INFO] Save path exists, checking model config...")
    loaded_model = load_model(args.save_path)
    if(loaded_model.get_config() == model.get_config()):
        print("[INFO] Model config is same, continuing training...")
        model = loaded_model
    else:
        print("[INFO] Model config different, training new model...")


model.fit(
    x_train, one_hot_train,
    epochs=5,
    batch_size=4096)
# endregion


# region    #! Evaluate Model
print("[INFO] Evaluating model...")
score = model.evaluate(x_test, one_hot_test, batch_size=1024)
print("[INFO] Test Accuracy: ")
print(score)
# endregion


# region    #! Saving Model
print("[INFO] Saving model...")
model.save(args.save_path)
# endregion


print("Process complete.")
