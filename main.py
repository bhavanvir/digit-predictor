from re import L
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
from tensorflow import keras

def view_image(v, x_test, y_test, predict, wrong_predict):
    fig, ax  = plt.subplots(1)

    ax.imshow(x_test[v].reshape(28, 28), cmap='BuPu')

    ax.set_title('Wrongly Predicted Image: ' + str(wrong_predict.index(v)) + '/' + str(len(wrong_predict)))

    legend = 'Predicted label: ' + str(np.argmax(predict[v])) + '\n' + 'Actual label: ' + str(y_test[v])
    ax.text(1, 14, legend, bbox={'facecolor': 'white', 'pad': 10})

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()
    
def find_incorrect(x_test, y_test, predict):
    wrong_predict = []
    for i in range(len(x_test)):
        if np.argmax(predict[i]) != y_test[i]:
            wrong_predict.append(i)
    
    return wrong_predict

def random_wrong_value(wrong_predict):
    v = random.choice(wrong_predict)

    return v

def create_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model

def new_model(x_train, y_train):
    model = create_model()

    model.summary()

    model.fit(x_train, y_train, epochs=50)
    model.save('model.h5')

    return model

def load_model(x_test, y_test):
    model = tf.keras.models.load_model("model.h5")

    model.summary()

    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    print("Restored model, loss: {:.2f}".format(loss))

    return model

def main():
    mnist_train = pd.read_csv(str(os.getcwd() + "\input\mnist_train.csv"), header=None)
    mnist_test = pd.read_csv(str(os.getcwd() + "\input\mnist_test.csv"), header=None)

    x_train = mnist_train.drop(0, axis=1).values 
    x_train = x_train / 255
    y_train = mnist_train[0].values 

    x_test = mnist_test.drop(0, axis=1).values
    x_test = x_test / 255
    y_test = mnist_test[0].values

    curr_dir = os.listdir(os.getcwd())
    if 'model.h5' in curr_dir:
        model = load_model(x_test, y_test)
    else:
        model = new_model(x_train, y_train)

    predict = model.predict(x_test)
    
    wrong_predict = find_incorrect(x_test, y_test, predict)
    v = random_wrong_value(wrong_predict)
    view_image(v, x_test, y_test, predict, wrong_predict)
    
if __name__ == "__main__":
    main()