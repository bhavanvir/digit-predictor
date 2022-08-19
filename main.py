import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
import PIL.ImageOps
from tensorflow import keras
from keras.datasets import mnist
from keras.optimizers import SGD
from PIL import Image 

def incorrect_image(v, x_test, y_test, predict, wrong_predict):
    fig, ax = plt.subplots(1)

    ax.imshow(x_test[v].reshape(28, 28), cmap='BuPu')
    ax.set_title('Wrongly Predicted Image: ' + str(wrong_predict.index(v)) + '/' + str(len(wrong_predict)))

    actual = np.where(y_test[v] == 1)
    legend = 'Predicted label: ' + str(np.argmax(predict[v])) + '\n' + 'Actual label: ' + str(actual[0][0])
    ax.text(x=1, y=25.9, s=legend, bbox={'facecolor': 'white', 'pad': 10})

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()

def external_image(img, file, predict):
    fig, ax = plt.subplots(1)
    
    ax.imshow(img.reshape(28, 28), cmap='BuPu')
    ax.set_title('Predicted File: ' + '\'' + str(file) + '\'')
    ax.text(x=1, y=25.9, s='Predicted label: ' + str(np.argmax(predict[0])), bbox={'facecolor': 'white', 'pad': 10})

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()

def external_data(file, input_dir):
    path = str(input_dir + '\\' + file)

    base_img = Image.open(path)
    black_white = base_img.convert("L")
    invert = PIL.ImageOps.invert(black_white)
    invert.save(input_dir + '\\' + file)

    img = tf.keras.preprocessing.image.load_img(path=path, color_mode='grayscale', target_size=(28, 28, 1))
    img = tf.keras.preprocessing.image.img_to_array(img)
    test_img = img.reshape((1, 784))

    return test_img, img

def mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

def find_all_incorrect(x_test, y_test, predict):
    wrong_predict = []
    
    for i in range(len(x_test)):
        index = np.where(y_test[i] == 1)
        if index[0][0] != np.argmax(predict[i]):
            wrong_predict.append(i)
    
    return wrong_predict

def random_incorrect(wrong_predict):
    v = random.choice(wrong_predict)

    return v

def create_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=SGD(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def new_model(x_train, y_train):
    model = create_model()

    model.summary()

    model.fit(x_train, y_train, epochs=1)
    model.save('mnist_model.h5')

    return model

def load_model(x_test, y_test):
    model = tf.keras.models.load_model("mnist_model.h5")

    model.summary()

    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    print("Restored model, loss: {:.2f}".format(loss))

    return model

def predict(model, x_test, y_test):
    print("\nExternal data must be placed in the 'input' folder.")
    query = input("Would you like to use your own external data or the MNIST data? (E/M): ")
    if query in ['E', 'e']:
        print()
        input_dir = str(os.getcwd() + '\input')
        curr_dir = os.listdir(input_dir)
      
        for file in curr_dir:
            test_img, img = external_data(file, input_dir)
            predict = model.predict(test_img)
            external_image(img, file, predict)
    elif query in ['M', 'm']:
        predict = model.predict(x_test)
        wrong_predict = find_all_incorrect(x_test, y_test, predict)
        v = random_incorrect(wrong_predict)
        incorrect_image(v, x_test, y_test, predict, wrong_predict)
    else:
        print("Error: invalid input, exiting...")
        exit(1)

def main():
    x_train, y_train, x_test, y_test = mnist_data()

    curr_dir = os.listdir(os.getcwd())
    if 'mnist_model.h5' in curr_dir:
        model = load_model(x_test, y_test)
    else:
        model = new_model(x_train, y_train)

    try:
        predict(model, x_test, y_test)
    except UnboundLocalError:
        print('Error: model must be trained before use, exiting...')
        
    
if __name__ == "__main__":
    main()