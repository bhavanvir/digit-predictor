# Paint GUI 
from locale import currency
import paint

# Operating System
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Input Formatting
from PyInquirer import style_from_dict, Token, prompt

# Image Processing
import cv2

# Math 
import math
import numpy as np 
import random

# Machine Learning
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

# Data Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Output Formatting
import colorama
colorama.init()
from termcolor import colored

# Time Tracking
import time

# String Validation
import re

global bolded_colour, colour
bolded_colour = '#FFFF00 bold'
colour = 'yellow'

global style 
style = style_from_dict({
    Token.Separator: bolded_colour,
    Token.QuestionMark: '',
    Token.Selected: bolded_colour,
    Token.Pointer: bolded_colour,
    Token.Instruction: '',
    Token.Answer: '',
    Token.Question: '',
})

def generate_external_image(img, file, y_predict, actual_label):
    fig, ax = plt.subplots(1)
    
    ax.imshow(img.reshape(28, 28), cmap='gray')
    ax.set_title('Image ' + '\'' + str(file) + '\'' + ' After Processing')

    legend = 'Predicted label: ' + str(np.argmax(y_predict[0])) + '\n' + 'Actual label: ' + actual_label
    ax.text(x=-10, y=8.02, s=legend, bbox={'facecolor': 'white', 'pad': 10})
    
    ax.text(x=-10, y=5.85, s=class_legend_label(y_predict, 0), bbox={'facecolor': 'white', 'pad': 10})

    zoom_plot_window()

    plt.show()

def generate_confusion_martix(y_predict, y_test):
    confusion_matrix = tf.math.confusion_matrix(labels=np.argmax(y_test, axis=1), predictions=np.argmax(y_predict, axis=1))

    heat_map = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='RdYlGn')
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=0)

    plt.title(label='Confusion Matrix for \'mnist_model.h5\'')
    plt.ylabel(ylabel='Predicted Label', fontsize=11)
    plt.xlabel(xlabel='Actual Label', fontsize=11)

    zoom_plot_window()

    plt.show()

def zoom_plot_window():
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')

def class_legend_label(y_predict, index):
    legend = ""
    start_index = 0
    for key, value in class_probabilities(y_predict, index).items():
        if start_index != 9:
            legend += 'Label: {}, Probability: {:.2f}%\n'.format(key, value * 100)
        else:
            legend += 'Label: {}, Probability: {:.2f}%'.format(key, value * 100)
        start_index += 1

    return legend

def image_composition(black_white_image, file):
    number_not_black = cv2.countNonZero(black_white_image)

    dimensions = black_white_image.shape
    size_difference = abs(dimensions[0] - dimensions[1])
    try:
        assert size_difference < 500
    except AssertionError:
        print(colored('Error: ' + '\'' + str(file) + '\'' ' has a dimensional difference that is ' + str(size_difference - 500) + ' pixels greater than the maximum, exiting...', color='red', attrs=['bold']))
        exit(1)

    height = black_white_image.shape[0]
    width = black_white_image.shape[1]

    number_pixels = height * width
    number_black = number_pixels - number_not_black  

    if number_black < number_not_black:
        return True
    else:
        return False

def rename_file(file):
    extension = re.search(r"[\.][a-zA-Z]*$", file)
    base_name = file.split(extension.group(0))[0]
    new_name = base_name + '_processed' + extension.group(0)

    return new_name

def process_image(file, base_path):
    new_name = rename_file(file)
    processed_path = str(os.getcwd() + '\processed_input\\' + new_name)

    base_image = cv2.imread(base_path)
    gray_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    (thresh, black_white_image) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if image_composition(black_white_image, file):
        inverted_image = cv2.bitwise_not(black_white_image)
    elif not image_composition(black_white_image, file):
        inverted_image = black_white_image

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dialated_image = cv2.dilate(inverted_image, kernel, iterations=15)

    blurred_image = cv2.GaussianBlur(dialated_image, (7, 7), 0)

    cv2.imwrite(processed_path, blurred_image)

    return processed_path

def find_all(x_test, y_test, y_predict):
    incorrect_predict, correct_predict = {} , {}
    
    for i in range(len(x_test)):
        index = np.where(y_test[i] == 1)
        if index[0][0] != np.argmax(y_predict[i]):
            incorrect_predict.update({i: np.argmax(y_predict[i])})
        else:
            correct_predict.update({i: np.argmax(y_predict[i])})
    
    return incorrect_predict, correct_predict

def test_harness(y_predict, file):
    extension = re.search(r"[\.][a-zA-Z]*$", file)
    removed_extension = file.strip(extension.group(0))
    
    test_case = re.search(r"[0-9]$", removed_extension)
    if str(np.argmax(y_predict[0])) == test_case.group(0):
        return test_case.group(0), True
    else:
        return test_case.group(0), False

def external_data(file, input_directory):
    base_path = str(input_directory + '\\' + file)
    processed_path = process_image(file, base_path)

    img = tf.keras.preprocessing.image.load_img(path=processed_path, color_mode='grayscale', target_size=(28, 28, 1))
    img = tf.keras.preprocessing.image.img_to_array(img)
    test_img = img.reshape(1, 28, 28, 1)
    test_img = test_img.astype('float32') / 255

    return test_img, img

def mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
        keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def data_transformation(x_train):
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
    )
    data_generator.fit(x_train)

    return data_generator

def new_model(x_train, y_train, x_test, y_test):
    model = create_model()
    data_generator = data_transformation(x_train)

    print(colored('\nSuccess: training new model...', color='green', attrs=['bold']))
    model.summary()

    early_stopping_monitor = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0,
        patience=50,
        verbose=0,
        mode='max',
        baseline=None,
        restore_best_weights=True
    )

    start = time.time()
    model.fit(data_generator.flow(x_train, y_train, batch_size=64), epochs=50, validation_data=(x_test, y_test), steps_per_epoch=x_train.shape[0] // 64, callbacks=[early_stopping_monitor])
    end = time.time()

    elapsed = end - start 
    hours = math.floor(elapsed / 3600)
    minutes = math.floor((elapsed - (hours * 3600)) / 60)
    seconds = math.floor(elapsed - (hours * 3600 + minutes * 60))
    print(colored('\nSuccess: model trained in {} hours, {} minutes, and {} seconds.'.format(hours, minutes, seconds), color='green', attrs=['bold']))

    model.save('mnist_model.h5')

    print(colored('Success: model saved as \'mnist_model.h5\' on disk, now exiting...', color='green', attrs=['bold']))
    exit(0)

def load_model(x_test, y_test):
    model = tf.keras.models.load_model("mnist_model.h5")

    print(colored('\nSuccess: loading \'mnist_model.h5\' from disk...', color='green', attrs=['bold']))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(colored('Success: restored \'mnist_model.h5\' with accuracy {:5.2f}%'.format(100 * accuracy) + ' and loss {:.2f}%.\n'.format(loss), color='green', attrs=['bold']))

    return model

def class_probabilities(y_predict, index):
    classes = {}
    for value in y_predict[index]:
        position = np.where(y_predict[index] == value)
        classes.update({position[0][0]: value})

    classes = {key: value for key, value in sorted(classes.items(), key=lambda item: item[1], reverse=True)}

    return classes

def external_data_query(model):
    input_directory = str(os.getcwd() + '\input')
    current_directory = os.listdir(input_directory)

    try:
        assert len(current_directory) > 0
    except AssertionError:
        print(colored('Error: no files found in \'input\' directory, exiting...', color='red', attrs=['bold']))
        exit(1)

    questions = [
        {
            'type': 'confirm',
            'qmark': '  ○',
            'name': 'processed',
            'message': 'Would you like to view the processed input image?',
            'default': 'invalid'
        }
    ]
    answers = prompt(questions, style=style)

    if answers['processed'] and answers['processed'] != 'invalid':
        view_flag = True
    elif not answers['processed'] and answers['processed'] != 'invalid':
        view_flag = False
    elif answers['processed'] == 'invalid':
        print(colored('Error: ' + '\'' + str(answers['processed']) + '\'' + ' is not in the correct format (Y/N), exiting...', color='red', attrs=['bold']))
        exit(1)

    sum = 0
    correct_files, incorrect_files = [], []
    for file in current_directory:
        test_img, img = external_data(file, input_directory)
        y_predict = model.predict(test_img)
        actual_label, boolean_label = test_harness(y_predict, file)
        
        try:
            if boolean_label:
                sum += 1
        except AttributeError:
            print(colored('Error: ' + '\'' + str(file) + '\'' + ' does not include a numerical label, exiting...', color='red', attrs=['bold']))
            exit(1)

        if actual_label == str(np.argmax(y_predict[0])):
            correct_files.append(file)
        else:
            incorrect_files.append(file)

        if view_flag:
            generate_external_image(img, file, y_predict, actual_label)
        else:
            print('  ○ File: ' + '\'' + str(file) + '\'')
            print('    ■ Predicted label: ' + str(np.argmax(y_predict[0])))
            print('    ■ Actual label: ' + actual_label)
            
    print('\n● Prediction summary:')
    print('  ○ Correctly predicted files: ' + str(correct_files)[1:-1])
    print('    ■ Percentage correct: {:.2f}% ({}/{})'.format((len(correct_files) / len(correct_files + incorrect_files)) * 100, len(correct_files), len(correct_files + incorrect_files)))
    print('  ○ Incorrectly predicted files: ' + str(incorrect_files)[1:-1])
    print('    ■ Percentage incorrect: {:.2f}% ({}/{})'.format((len(incorrect_files) / (len(correct_files + incorrect_files))) * 100, len(incorrect_files), len(correct_files + incorrect_files)))

def prediction_query(model, x_test, y_test):
    print(colored('\n● Note:', color='yellow'))
    print(colored('  ○ External data must be placed in the \'input\' folder.', color='yellow'))
    print(colored('  ○ For testing purposes, have the input file include the number wanting to be predicted.\n', color='yellow'))

    questions = [
        {
            'type': 'list',
            'qmark': '●',
            'name': 'query_type',
            'message': 'How would you like to query the model?',
            'choices': [
                {
                    'name': 'Upload files',
                    'value': 'external'
                },
                {
                    'name': 'Paint digits',
                    'value': 'drawn'
                }
            ]
        }
    ]
    answers = prompt(questions, style=style)

    if answers['query_type'] == 'external':
        create_directory()
        external_data_query(model)
    elif answers['query_type'] == 'drawn':
        paint.main()

def confusion_matrix_query(model, x_test, y_test):
    questions = [
        {
            'type': 'confirm',
            'qmark': '●',
            'name': 'confusion_matrix',
            'message': 'Would you like to view the confusion matrix?',
            'default': 'invalid'
        }
    ]
    answers = prompt(questions, style=style)

    if answers['confusion_matrix'] and answers['confusion_matrix'] != 'invalid':
        y_predict = model.predict(x_test)
        generate_confusion_martix(y_test, y_predict)
    elif not answers['confusion_matrix'] and answers['confusion_matrix'] != 'invalid':
        return
    elif answers['confusion_matrix'] == 'invalid':
        print(colored('Error: ' + '\'' + str(answers['confusion_matrix']) + '\'' + ' is not in the correct format (Y/N), exiting...', color='red', attrs=['bold']))
        exit(1)

def create_directory():
    try:
        os.mkdir(os.getcwd() + '/processed_input')
    except FileExistsError:
        pass

def delete_directory():
    try:
        current_directory = os.listdir(os.getcwd() + '/processed_input')
        for file in current_directory:
            extension = re.search(r"[\.][a-zA-Z]*$", file)
            if (extension.group(0)).lower() in ['.png', '.jpg', '.jpeg']:
                os.remove(os.getcwd() + '/processed_input/' + file)
        os.rmdir(os.getcwd() + '/processed_input')
    except FileNotFoundError:
        pass

def delete_pycache():
    try:
        current_directory = os.listdir(os.getcwd() + '/__pycache__')
        for file in current_directory:
            os.remove(os.getcwd() + '/__pycache__/' + file)
        os.rmdir(os.getcwd() + '/__pycache__')
    except FileNotFoundError:
        pass

def main():
    x_train, y_train, x_test, y_test = mnist_data()

    current_directory = os.listdir(os.getcwd())
    if 'mnist_model.h5' in current_directory:
        model = load_model(x_test, y_test)
    else:
        model = new_model(x_train, y_train, x_test, y_test)

    confusion_matrix_query(model, x_test, y_test)

    try:
        prediction_query(model, x_test, y_test)
    except UnboundLocalError:
        print(colored("Error: model must be trained before use, exiting...", color='red', attrs=['bold']))

    delete_directory()
    delete_pycache()
        
if __name__ == "__main__":
    main()