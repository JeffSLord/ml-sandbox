# region     #! Imports
import argparse
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model, Model
from keras.callbacks import TensorBoard
from keras.applications.inception_v3 import InceptionV3
import numpy as np
import os
from time import time
# endregion
file_path, file_name = os.path.split(__file__)
print(file_path)

# region     #! Arguments
parser = argparse.ArgumentParser(description='Argument parser.')
parser.add_argument('--dir', '-d', help='Path to root directory')
parser.add_argument('--input_shape', '-i', default=299, type=int)
parser.add_argument('--color_scale', '-c', default=3, type=int)
parser.add_argument('--epochs', '-e', default=10, type=int)
parser.add_argument('--batch_size', '-b', default=32, type=int)
parser.add_argument('--validation_split', '-v', default=0.2)
parser.add_argument('--save_path', '-s',
                    default=os.path.join(file_path, "k_inception_model.h5"))
parser.add_argument('--log_dir', '-log',
                    default=os.path.join(file_path, "logs/"))
parser.add_argument('--load_model', '-l', default=None)
#? ...
args = parser.parse_args()
# endregion


# region    #! Process Data
print("[INFO] Processing data...")
gen = ImageDataGenerator(
    rescale=1./255,         # this is for rgb colors
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=args.validation_split,
)
training_iterator = gen.flow_from_directory(
    args.dir,
    target_size=(args.input_shape, args.input_shape),
    batch_size=args.batch_size,
    subset='training',  # other option is 'validation'
)
eval_iterator = gen.flow_from_directory(
    args.dir,
    target_size=(args.input_shape, args.input_shape),
    batch_size=args.batch_size,
    shuffle=False,
    subset='validation'
)
print("[INFO] Inferred classes: " + str(training_iterator.class_indices))
print("[INFO] Length of classes: " + str(len(training_iterator.class_indices)))
# endregion


# region    #! Define Model
print("[INFO] Defining model...")
base_model = InceptionV3(include_top=False)
print(base_model.output)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(training_iterator.class_indices),
                    activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
# endregion


# region    #! Compile Model
print("[INFO] Compiling model...")
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.RMSprop(),
    metrics=['accuracy']
)
# endregion


# region    #! Train Model
print("[INFO] Training model...")
if(args.load_model != None and os.path.isfile(args.load_model)):
    print("[INFO] Continuing training from loaded model...")
    model = load_model(args.load_model)
elif(os.path.isfile(args.save_path)):
    print("[INFO] Save path exists, continuing training...")
    model = load_model(args.save_path)

model.fit_generator(
    training_iterator,
    epochs=args.epochs,
    verbose=1
)
# endregion


# region    #! Evaluate Model
print("[INFO] Evaluating model...")
eval_result = model.evaluate_generator(
    eval_iterator
)
for i in range(len(model.metrics_names)):
    print("[INFO] " + model.metrics_names[i] +
          ":% .2f %%" % (eval_result[i]*100))
# endregion


# predict_result = model.predict_generator(eval_iterator)
# predict_result_binary = np.argmax(predict_result, axis=-1)
# print(eval_iterator.filenames)
# print(predict_result_binary)


# region    #! Saving Model
print("[INFO] Saving model...")
model.save(args.save_path)
# endregion

print("Process complete.")
