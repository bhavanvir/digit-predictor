from tensorflow import keras
from keras.datasets import mnist
from keras.optimizers import SGD
from termcolor import colored
import cv2
import re
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
import colorama
import time

colorama.init()

def generate_image(v, x_test, y_test, y_predict, occurences):
    fig, ax = plt.subplots(1)

    ax.imshow(x_test[v].reshape(28, 28), cmap='gray')
    ax.set_title('Image Number ' + str(occurences.index(v)) + ' of ' + str(len(occurences)) + ' Total Occurences')

    actual = np.where(y_test[v] == 1)
    legend = 'Predicted label: ' + str(np.argmax(y_predict[v])) + '\n' + 'Actual label: ' + str(actual[0][0])
    ax.text(x=1, y=25.9, s=legend, bbox={'facecolor': 'white', 'pad': 10})

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()

def generate_external_image(img, file, y_predict):
    fig, ax = plt.subplots(1)
    
    ax.imshow(img.reshape(28, 28), cmap='gray')
    ax.set_title('Image ' + '\'' + str(file) + '\'' + ' After Processing')
    ax.text(x=1, y=25.9, s='Predicted label: ' + str(np.argmax(y_predict[0])), bbox={'facecolor': 'white', 'pad': 10})

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()

def generate_random_image(x_test, y_test, y_predict, prediction, number_range):
    occurence = find_occurences(prediction, int(number_range))
    v = random_predict(occurence)
    generate_image(v, x_test, y_test, y_predict, occurence)

def process_image(base_path, processed_path):
    base_img = cv2.imread(base_path)

    gray_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    (thresh, black_white_img) = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inverted_img = cv2.bitwise_not(black_white_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(inverted_img, kernel, iterations=5)

    cv2.imwrite(processed_path, dilation)

def find_all(x_test, y_test, y_predict):
    incorrect_predict, correct_predict = {} , {}
    
    for i in range(len(x_test)):
        index = np.where(y_test[i] == 1)
        if index[0][0] != np.argmax(y_predict[i]):
            incorrect_predict.update({i: np.argmax(y_predict[i])})
        else:
            correct_predict.update({i: np.argmax(y_predict[i])})
    
    return incorrect_predict, correct_predict

def random_predict(occurences):
    v = random.choice(occurences)

    return v

def test_harness(y_predict, file):
    test_case = re.search(r"[0-9]", file)
    if str(np.argmax(y_predict[0])) == test_case.group(0):
        return True
    else:
        return False

def find_occurences(prediction, wanted):
    occurences = [k for k, v in prediction.items() if v == wanted]
    
    return occurences

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
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

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

def new_model(x_train, y_train, x_test, y_test):
    model = create_model()

    print(colored('\nSuccess: training new model...', color='green', attrs=['bold']))
    model.summary()

    start = time.time()
    model.fit(x_train, y_train, batch_size=64, epochs=1000, validation_data=(x_test, y_test))
    end = time.time()
    print(colored('Success: model trained in ' + '{:.2f}'.format(end - start) + ' seconds.', color='green', attrs=['bold']))

    model.save('mnist_model.h5')

    print(colored('Success: model saved as \'mnist_model.h5\' on disk, now exiting...', color='green', attrs=['bold']))
    exit(0)

def load_model(x_test, y_test):
    model = tf.keras.models.load_model("mnist_model.h5")

    print(colored('\nSuccess: loading \'mnist_model.h5\' from disk...', color='green', attrs=['bold']))
    model.summary()

    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(colored('Success: restored \'mnist_model.h5\' with accuracy {:5.2f}%'.format(100 * acc) + ' and loss {:.2f}'.format(loss), color='green', attrs=['bold']))

    return model

def external_data_query(model):
    input_dir = str(os.getcwd() + '\input')
    curr_dir = os.listdir(input_dir)

    try:
        assert len(curr_dir) > 0
    except AssertionError:
        print(colored('Error: no files found in \'input\' directory, exiting...', color='red', attrs=['bold']))
        exit(1)

    view_img = input("● Would you like to view the processed input image? (Y/N): ")
    if view_img in ['Y', 'y']:
        view_flag = True
    elif view_img in ['N', 'n']:
        view_flag = False
    else:
        print(colored('Error: ' + '\'' + str(view_img) + '\'' + ' is not in the correct format (Y/N), exiting...', color='red', attrs=['bold']))
        exit(1)

    sum = 0
    for file in curr_dir:
        print()
        test_img, img = external_data(file, input_dir)
        y_predict = model.predict(test_img)
        
        try:
            if test_harness(y_predict, file):
                sum += 1
        except AttributeError:
            print(colored('Error: ' + '\'' + str(file) + '\'' + ' does not include a numerical label (0-9), exiting...', color='red', attrs=['bold']))
            exit(1)

        if view_flag:
            generate_external_image(img, file, y_predict)
        elif not view_flag:
            print('File: ' + '\'' + str(file) + '\'' + '\nPrediction: ' + str(np.argmax(y_predict[0])))
    print(colored('\nPrediction summary: ' + str(sum) + '/' + str(len(curr_dir)) + ' or ' + str(sum / (len(curr_dir)) * 100) + '%' + ' are correct.', color='yellow'))

def mnist_data_query(model, x_test, y_test):
    incorrect_correct = input("● Would you like to see an incorrectly predicted image or a correctly predicted image? (I/C): ")
    try:
        assert incorrect_correct in ['I', 'i', 'C', 'c']
    except AssertionError:
        print(colored('Error: ' + '\'' + str(incorrect_correct) + '\'' + ' is not in the correct format (I/C), exiting...', color='red', attrs=['bold']))
        exit(1)

    number_range = input("  ○ Enter a number of the image you would like to see (0-9): ")
    print()
    try:
        assert int(number_range) in range(10)
    except AssertionError:
        print(colored('Error: ' + '\'' + str(number_range) + '\'' + ' is not within range (0-9), exiting...', color='red', attrs=['bold']))
        exit(1)

    y_predict = model.predict(x_test)
    incorrect_predict, correct_predict = find_all(x_test, y_test, y_predict)
    if incorrect_correct in ['I', 'i']:
        generate_random_image(x_test, y_test, y_predict, incorrect_predict, number_range)
    elif incorrect_correct in ['C', 'c']:
        generate_random_image(x_test, y_test, y_predict, correct_predict, number_range)

def prediction_query(model, x_test, y_test):
    print(colored('\n● Note:', color='yellow'))
    print(colored('  ○ External data must be placed in the \'input\' folder.', color='yellow'))
    print(colored('  ○ For testing purposes, have the input file include the number wanting to be predicted.\n', color='yellow'))

    external_mnist = input("● Would you like to use your own external data or the MNIST data? (E/M): ")
    if external_mnist in ['E', 'e']:
        external_data_query(model)
    elif external_mnist in ['M', 'm']:
        mnist_data_query(model, x_test, y_test)
    else:
        print(colored('Error: ' + '\'' + str(external_mnist) + '\'' + ' is not in the correct format (E/M), exiting...', color='red', attrs=['bold']))
        exit(1)

def main():
    x_train, y_train, x_test, y_test = mnist_data()

    curr_dir = os.listdir(os.getcwd())
    if 'mnist_model.h5' in curr_dir:
        model = load_model(x_test, y_test)
    else:
        model = new_model(x_train, y_train, x_test, y_test)

    try:
        prediction_query(model, x_test, y_test)
    except UnboundLocalError:
        print(colored("Error: model must be trained before use, exiting...", color='red', attrs=['bold']))
        
if __name__ == "__main__":
    main()