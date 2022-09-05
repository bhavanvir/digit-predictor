# Operating System
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GUI
from tkinter import *
from tkinter import colorchooser

# Image Processing
from PIL import ImageGrab
import cv2

# Math
import numpy as np

# Machine Learning
import logging
import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Time Delay
import time

# Regular Expression
import re

class Paint(object):
    DEFAULT_PEN_SIZE = 10.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()
        self.root.title("Paint")
        self.root.resizable(False, False)

        self.centre_window()

        menu_bar = Menu(self.root)

        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Predict", command=self.predict_shortcut, accelerator="P")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit_shortcut, accelerator="Ctrl+Q")
        menu_bar.add_cascade(label="File", menu=file_menu)

        colour_menu = Menu(menu_bar, tearoff=0)
        colour_menu.add_command(label="Brush Colour", command=self.change_foreground, accelerator="F")
        colour_menu.add_command(label="Background Colour", command=self.change_background, accelerator="B")
        colour_menu.add_separator()
        colour_menu.add_command(label="Clear Canvas", command=self.clear_shortcut, accelerator="C")
        menu_bar.add_cascade(label="Edit", menu=colour_menu)

        self.root.config(menu=menu_bar)

        self.canvas = Canvas(self.root, bg='white', width=600, height=600)
        self.canvas.grid(row=1, columnspan=5)

        self.root.bind_all("<p>", self.predict_shortcut)
        self.root.bind_all("<c>", self.clear_shortcut)
        self.root.bind_all("<Control-q>", self.exit_shortcut)
        self.root.bind_all("<f>", self.change_foreground_shortcut)
        self.root.bind_all("<b>", self.change_background_shortcut)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.colour_background = 'white'
        self.colour_foreground = 'black'
        self.color = self.DEFAULT_COLOR
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
    
    def change_foreground_shortcut(self, event):
        self.change_foreground()

    def change_background_shortcut(self, event):
        self.change_background()

    def exit_shortcut(self, event):
        self.root.destroy()
    
    def predict_shortcut(self, event):
        self.screenshot_canvas()
    
    def clear_shortcut(self, event):
        self.clear()

    def screenshot_canvas(self):
        time.sleep(0.25)
        x0 = self.root.winfo_rootx()
        y0 = self.root.winfo_rooty()
        x1 = x0 + self.root.winfo_width()
        y1 = y0 + self.root.winfo_height()

        image = ImageGrab.grab().crop((x0, y0, x1, y1))
        image.save(os.getcwd() + '/screenshot/screenshot.png')
        self.predict_drawing()

    def image_composition(self, black_white_image):
        number_not_black = cv2.countNonZero(black_white_image)

        height = black_white_image.shape[0]
        width = black_white_image.shape[1]

        number_pixels = height * width
        number_black = number_pixels - number_not_black  

        if number_black < number_not_black:
            return True
        else:
            return False
    
    def process_image(self):
        file = os.listdir(os.getcwd() + '\screenshot')[0]
        base_path = (os.getcwd() + '\screenshot\\' + file)

        new_name = rename_file(file)
        processed_path = str(os.getcwd() + '\processed_screenshot\\' + new_name)

        base_image = cv2.imread(base_path)
        gray_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        (thresh, black_white_image) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if self.image_composition(black_white_image):
            inverted_image = cv2.bitwise_not(black_white_image)
        elif not self.image_composition(black_white_image, file):
            inverted_image = black_white_image

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dialated_image = cv2.dilate(inverted_image, kernel, iterations=10)

        blurred_image = cv2.GaussianBlur(dialated_image, (7, 7), 0)

        cv2.imwrite(processed_path, blurred_image)

        return processed_path

    def load_model(self):
        model = tf.keras.models.load_model("mnist_model.h5")

        return model

    def class_probabilities(self, y_predict, index):
        classes = {}
        for value in y_predict[index]:
            position = np.where(y_predict[index] == value)
            classes.update({position[0][0]: str('{:.2f}'.format(value * 100))})

        return classes

    def predict_drawing(self):
        model = self.load_model()
        processed_path = self.process_image()

        img = tf.keras.preprocessing.image.load_img(path=processed_path, color_mode='grayscale', target_size=(28, 28, 1))
        img = tf.keras.preprocessing.image.img_to_array(img)
        test_img = img.reshape(1, 28, 28, 1)
        test_img = test_img.astype('float32') / 255

        y_predict = model.predict(test_img)
        classes = self.class_probabilities(y_predict, 0)
        print("  ○ Label: {}, Probability: {}%".format(str(np.argmax(y_predict[0])), classes[np.argmax(y_predict[0])]))

    def centre_window(self, window_height=600, window_width=600):
        self.root.resizable(False, False)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        self.root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

    def clear(self):
        self.canvas.delete("all")
    
    def change_foreground(self):
        self.colour_foreground=colorchooser.askcolor(color=self.colour_foreground)[1]

    def change_background(self):
        self.colour_background=colorchooser.askcolor(color=self.colour_background)[1]
        self.canvas['bg'] = self.colour_background

    def paint(self, event):
        self.line_width = self.DEFAULT_PEN_SIZE
        paint_color = self.colour_foreground

        if self.old_x and self.old_y:
            self.canvas.create_line(
                self.old_x, 
                self.old_y, 
                event.x, 
                event.y,
                width=self.line_width, 
                fill=paint_color,
                capstyle=ROUND, 
                smooth=TRUE, 
                splinesteps=36
            )

        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

def create_directory(name):
        try:
            os.mkdir(os.getcwd() + '/' + name)
        except FileExistsError:
            pass
    
def delete_directory(name):
    try:
        current_directory = os.listdir(os.getcwd() + '/' + name)
        for file in current_directory:
            extension = re.search(r"[\.][a-zA-Z]*$", file)
            if (extension.group(0)).lower() in ['.png', '.jpg', '.jpeg']:
                os.remove(os.getcwd() + '/' + name + '/' + file)
        os.rmdir(os.getcwd() + '/' + name)
    except FileNotFoundError:
        pass

def rename_file(file):
    extension = re.search(r"[\.][a-zA-Z]*$", file)
    base_name = file.split(extension.group(0))[0]
    new_name = base_name + '_processed' + extension.group(0)

    return new_name

def main():
    create_directory('screenshot')
    create_directory('processed_screenshot')
    Paint()
    delete_directory('screenshot')
    delete_directory('processed_screenshot')

if __name__ == '__main__':
    main()