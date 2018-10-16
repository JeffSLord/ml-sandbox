from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle

# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

### Export Test ###

# Set up model structure
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
model = tflearn.DNN(network, tensorboard_verbose=0,
                    checkpoint_path='bird-classifier.tfl.ckpt')

# Load
model.load("bird-classifier.tfl")
builder = tf.saved_model.builder.SavedModelBuilder("./SavedModelForServing")
signature = tf.saved_model.signature_def_utils.predict_signature_def(
    inputs={'in': input_data(shape=[None, 32, 32, 3],
                             data_preprocessing=img_prep,
                             data_augmentation=img_aug)},
    outputs={'out': network})
builder.add_meta_graph_and_variables(model.session,
                                     [tf.saved_model.tag_constants.SERVING],
                                     signature_def_map={'serving_default': signature})

builder.save()
