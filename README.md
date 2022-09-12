# Digit-Predictor

Digit-Predictor is a command-line, convolutional neural network that is able to classify handwritten digits. 

The [MNIST](https://keras.io/api/datasets/mnist/) digits classification dataset was used for training; the dataset consists of 60,000 28x28 grayscale images of 10 digits, along with a test set of 10,000 images for validation during training.

## Installation

The source code for Digit-Predictor can then be obtained using the `git clone` command with the public repository's URL as the target, to make a clone or copy of the repository in a new directory, at another location.
```bash
git clone https://github.com/bhavanvir/Digit-Predictor
```

Change your directory to the root of the project.
```bash
cd Digit-Predictor
```

Several dependencies are required to run Digit-Predictor that are not included in the Python Standard Library. It is imperative that these modules are installed and functional beforehand.

Using the package manager [pip](https://pip.pypa.io/en/stable/) to install all external modules:
```bash
pip install -r requirements.txt
```

## Usage

To use the built-in Digit-Predictor test harness, have the digit wanting to be predicted successfully in the name of the file. For example, if the image is of digit 3, the file name should be `test_3.jpg` or something similar.

If the `mnist_model.h5` file is not present in the root directory, the Digit-Predictor application will first run a training session to generate the model; the default epoch length is defined as `50`, but this can be changed by altering the `epochs` parameter within the `new_model` function.
```python
model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=50, validation_data=(x_test, y_test), steps_per_epoch=x_train.shape[0] // 64, callbacks=[early_stopping_monitor])
```

Running the Digit-Predictor application using `python3 main.py` in any terminal will prompt the user with three branching options: `MNIST` to use the MNIST data, `External` to use your own external data, or `Drawn` to draw your own data. 

1. If `Upload files` is selected, the user will be prompted with a `Y` or `N` option to view their processed input image, or to recieve a command-line prediction for each file located in the `input` directory. In either case, a prediction summary will be output to the terminal.
2. If `Paint digits` is selected, the user will be prompted with a paint canvas that can be used to draw a digit, then when they are satisfied with their drawing, they can select `File → Predict` or use the keyboard short-cut `P`, to recieve a prediction.

Users can also select either `Y` or `N` during application start-up, to view the confusion matrix associated with the `mnist_model.h5` file; the confusion matrix serves as a visual representation of the accuracy of the model.

## License
[MIT](https://choosealicense.com/licenses/mit/)