import cv2
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
import colorama
from tensorflow import keras
from keras.datasets import mnist
from keras.optimizers import SGD
from termcolor import colored

colorama.init()

def generate_image(v, x_test, y_test, predict, occurences):
    fig, ax = plt.subplots(1)

    ax.imshow(x_test[v].reshape(28, 28), cmap='gray')
    ax.set_title('Image Number ' + str(occurences.index(v)) + ' of ' + str(len(occurences)) + ' Total Occurences')

    actual = np.where(y_test[v] == 1)
    legend = 'Predicted label: ' + str(np.argmax(predict[v])) + '\n' + 'Actual label: ' + str(actual[0][0])
    ax.text(x=1, y=25.9, s=legend, bbox={'facecolor': 'white', 'pad': 10})

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()

def find_all(x_test, y_test, predict):
    incorrect_predict = {}
    correct_predict = {}
    
    for i in range(len(x_test)):
        index = np.where(y_test[i] == 1)
        if index[0][0] != np.argmax(predict[i]):
            incorrect_predict.update({i: np.argmax(predict[i])})
        else:
            correct_predict.update({i: np.argmax(predict[i])})
    
    return incorrect_predict, correct_predict

def random_predict(predict):
    v = random.choice(predict)

    return v

def find_occurences(prediction, wanted):
    occurences = [k for k, v in prediction.items() if v == wanted]
    
    return occurences

def external_image(img, file, predict):
    fig, ax = plt.subplots(1)
    
    ax.imshow(img.reshape(28, 28), cmap='gray')
    ax.set_title('Image ' + '\'' + str(file) + '\'' + ' After Processing')
    ax.text(x=1, y=25.9, s='Predicted label: ' + str(np.argmax(predict[0])), bbox={'facecolor': 'white', 'pad': 10})

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()

def process_image(base_path, processed_path):
    base_img = cv2.imread(base_path)
    gray_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    (thresh, black_white_img) = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    inverted_img = cv2.bitwise_not(black_white_img)
    cv2.imwrite(processed_path, inverted_img)

def external_data(file, input_dir):
    base_path = str(input_dir + '\\' + file)
    processed_path = str(os.getcwd() + '\processed_input\\' + file)

    process_image(base_path, processed_path)

    img = tf.keras.preprocessing.image.load_img(path=processed_path, color_mode='grayscale', target_size=(28, 28, 1))
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

def create_model():
    model = keras.Sequential([
        keras.layers.Dense(784, activation='relu', input_shape=(784,)),
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

    print(colored('\nSuccess: training new model...', color='green', attrs=['bold']))
    model.summary()

    model.fit(x_train, y_train, epochs=500)
    model.save('mnist_model.h5')

    return model

def load_model(x_test, y_test):
    model = tf.keras.models.load_model("mnist_model.h5")

    print(colored('\nSuccess: loading \'mnist_model.h5\' from disk...', color='green', attrs=['bold']))
    model.summary()

    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(colored('Success: restored \'mnist_model.h5\' with accuracy {:5.2f}%'.format(100 * acc) + ' and loss {:.2f}'.format(loss), color='green', attrs=['bold']))

    return model

def predict(model, x_test, y_test):
    print("\nExternal data must be placed in the 'input' folder.")
    external_mnist = input("● Would you like to use your own external data or the MNIST data? (E/M): ")
    if external_mnist in ['E', 'e']:
        input_dir = str(os.getcwd() + '\input')
        curr_dir = os.listdir(input_dir)

        if len(curr_dir) > 1:
            view_img = input("● Would you like to view the processed input images? (Y/N): ")
        else:
            view_img = input("● Would you like to view the processed input image? (Y/N): ")

        if view_img in ['Y', 'y']:
            view_flag = True
        elif view_img in ['N', 'n']:
            view_flag = False
        else:
            print(colored('Error: ' + '\'' + str(view_img) + '\'' + ' is not in the correct format (Y/N), exiting...', color='red', attrs=['bold']))
            exit(1)
      
        for file in curr_dir:
            print()
            test_img, img = external_data(file, input_dir)
            predict = model.predict(test_img)
            if view_flag:
                external_image(img, file, predict)
            elif not view_flag:
                print('File: ' + '\'' + str(file) + '\'' + '\nPrediction: ' + str(np.argmax(predict[0])))
    elif external_mnist in ['M', 'm']:
        incorrect_correct = input("● Would you like to see an incorrectly predicted image or a correctly predicted image? (I/C): ")

        number_range = input("  ○ Enter the number of the image you would like to see (0-9): ")
        try:
            assert int(number_range) in range(10)
        except AssertionError:
            print(colored('Error: ' + '\'' + str(number_range) + '\'' + ' is not within range (0-9), exiting...', color='red', attrs=['bold']))
            exit(1)

        predict = model.predict(x_test)
        incorrect_predict, correct_predict = find_all(x_test, y_test, predict)
        if incorrect_correct in ['I', 'i']:
            incorrect_occur = find_occurences(incorrect_predict, int(number_range))
            v = random_predict(incorrect_occur)
            generate_image(v, x_test, y_test, predict, incorrect_occur)
        elif incorrect_correct in ['C', 'c']:
            correct_occur = find_occurences(correct_predict, int(number_range))
            v = random_predict(correct_occur)
            generate_image(v, x_test, y_test, predict, correct_occur)
    else:
        print(colored('Error: ' + '\'' + str(external_mnist) + '\'' + ' is not in the correct format (E/M), exiting...', color='red', attrs=['bold']))
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
        print(colored("Error: model must be trained before use, exiting...", color='red', attrs=['bold']))
        
if __name__ == "__main__":
    main()