# region     #! Imports
import argparse
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Conv1D, MaxPooling1D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.callbacks import TensorBoard
import numpy as np
import os
from time import time

# endregion
file_path, file_name = os.path.split(__file__)

# region     #! Arguments
parser = argparse.ArgumentParser(description='Argument parser.')
parser.add_argument('--dir', '-d', help='Path to root directory')
parser.add_argument('--input_shape', '-i', default=299, type=int)
parser.add_argument('--color_scale', '-c', default=3, type=int)
parser.add_argument('--epochs', '-e', default=10, type=int)
parser.add_argument('--batch_size', '-b', default=32, type=int)
parser.add_argument('--validation_split', '-v', default=0.2)
parser.add_argument('--save_path', '-s',
                    default=os.path.join(file_path, "k_image_model.h5"))
parser.add_argument('--log_dir', '-log',
                    default=os.path.join(file_path, "logs/"))
parser.add_argument('--load_model', '-l', default=None)
#? ...
args = parser.parse_args()
# endregion


# region    #! Process Data
print("[INFO] Processing data...")
train_gen = ImageDataGenerator(
    rescale=1./255,         # this is for rgb colors?
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=args.validation_split,
)

train_iterator = train_gen.flow_from_directory(
    args.dir,
    target_size=(args.input_shape, args.input_shape),
    batch_size=args.batch_size,
    shuffle=True,
    subset='training',  # other option is 'validation'
)
validation_iterator = train_gen.flow_from_directory(
    args.dir,
    target_size=(args.input_shape, args.input_shape),
    batch_size=args.batch_size,
    shuffle=True,
    subset='validation'
)
predict_iterator = train_gen.flow_from_directory(
    args.dir,
    target_size=(args.input_shape, args.input_shape),
    batch_size=args.batch_size,
    shuffle=False,
    subset='validation'
)
print("[INFO] Inferred classes: " + str(train_iterator.class_indices))
print("[INFO] Length of classes: " + str(len(train_iterator.class_indices)))
# endregion


# region    #! Define Model
print("[INFO] Defining model...")
model = Sequential()
model.add(Conv2D(
    32, kernel_size=(3, 3), strides=(1, 1),
    activation='relu',
    input_shape=(args.input_shape, args.input_shape, args.color_scale)
))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_iterator.class_indices), activation='softmax'))
# endregion


# region    #! Compile Model
print("[INFO] Compiling model...")
model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.RMSprop(),
    metrics=['accuracy']
)
tensorboard = TensorBoard(log_dir=args.log_dir+"{}".format(time()))
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

model.fit_generator(
    train_iterator,
    validation_data=validation_iterator,
    epochs=args.epochs,
    verbose=1
)
# endregion


# region    #! Evaluate Model
print("[INFO] Evaluating model...")
eval_result = model.evaluate_generator(
    predict_iterator
)
for i in range(len(model.metrics_names)):
    print("[INFO] " + model.metrics_names[i] +
          ":% .2f %%" % (eval_result[i]*100))
# endregion


predict_result = model.predict_generator(predict_iterator)
# print(predict_result)
predict_result_binary = np.argmax(predict_result, axis=-1)
# print(eval_iterator.filenames)
print(predict_result_binary)


# region    #! Saving Model
print("[INFO] Saving model...")
model.save(args.save_path)
# endregion

print("Process complete.")
