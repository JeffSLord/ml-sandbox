import tensorflow as tf
import keras
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split

file_path = os.path.abspath(__file__)
file_path = os.path.dirname(os.path.dirname(file_path))


def input_fn(x_data, y_data=None):
    y_data = np.array(y_data)
    x_data = tf.constant(x_data)
    y_data = tf.constant(y_data)


# region    #! Process Data
def process():
    x = []
    y = []

    for i in range(10000):
        ax = random.randint(1, 100)
        bx = random.randint(1, 100)
        x.append([ax, bx])
        if(ax+bx < 100):
            y.append(0)
        else:
            y.append(1)

    x_data = np.array(x)
    y_data = np.array(y)
    print(x_data.shape)
    print(y_data.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2)
    # x_data_tf = tf.constant(x_data)
    # y_data_tf = tf.constant(y_data)
    # dataset = tf.data.Dataset.from_tensor_slices((x_data_tf, y_data_tf))
    # print(dataset)
    print(x_train.shape)
    print(y_train.shape)
    # endregion

    # iterator = dataset.make_one_shot_iterator()
    # with tf.Session() as sess:
    #     a = iterator.get_next()
    #     print(a[0][0])

    feature_columns = [tf.feature_column.numeric_column(
        "x",
        shape=2,
        dtype=tf.float32
    )]
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=y_train,
        num_epochs=None,
        shuffle=True
    )
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False
    )

    model_dir = os.path.join(file_path, "models", "estimator", "test")
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[
            10, 20, 10],
        n_classes=2,
        optimizer=tf.train.AdagradOptimizer(
            learning_rate=0.1
        ),
        activation_fn=tf.nn.relu,
        model_dir=model_dir
    )

    print("[INFO] Beginning training...")
    classifier.train(input_fn=train_input_fn, steps=2000)
    print("[INFO] Training complete.")

    accuracy_score = classifier.evaluate(
        input_fn=test_input_fn)["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
    return accuracy_score


def main():
    for i in range(10):
        process()


if __name__ == "__main__":
    main()
