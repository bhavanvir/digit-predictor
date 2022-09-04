# Digit-Predictor

Digit-Predictor is a command-line, convolutional neural network that is able to classify handwritten digits. 

The [MNIST](https://keras.io/api/datasets/mnist/) digits classification dataset was used for training; the dataset consists of 60,000 28x28 grayscale images of 10 digits, along with a test set of 10,000 images for validation during training.

## Installation

Several dependencies are required to run Digit-Predictor that are not included in the Python Standard Library. It is imperative that these modules are installed and functional beforehand.

Using the package manager [pip](https://pip.pypa.io/en/stable/) to install external modules:
```bash
pip install cv2
pip install numpy 
pip install tensorflow
pip install keras
pip install matplotlib
pip install seaborn
pip install colorama
```

## Usage

To use the built-in Digit-Predictor test harness, have the digit wanting to be predicted successfully in the name of the file. For example, if the image is of digit 3, the file name should be `test_3.jpg` or something similar.

If the `mnist_model.h5` file is not present in the root directory, the Digit-Predictor application will first run a training session to generate the model; the default epoch length is defined as `50`, but this can be changed by altering the `epochs` parameter within the `new_model` function.
```python
model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=50, validation_data=(x_test, y_test), steps_per_epoch=x_train.shape[0] // 64, callbacks=[early_stopping_monitor])
```

Running the Digit-Predictor application using `python3 main.py` in any terminal will prompt the user with three branching options: `E` to use your own external data, `M` to use the MNIST data, or `D` to draw your own data. 

1. If `E` is selected, the user will be prompted with the option to view the processed version of their input image, or to recieve a command-line prediction along with each image read from the `input` directory.
2. If `M` is selected, the user will be prompted with the option to view an incorrectly predicted image, or a correctly predicted image. In either case, the user will then be asked to input a digit between 0 and 9 inclusive, and the application will then show an example.
3. If `D` is selected, the user will be prompted with a paint canvas that can be used to draw a digit, then when they are satisfied with their drawing, they can select `File → Predict` to recieve a prediction.

Users can also select either `Y` or `N` during application start-up, to view the confusion matrix associated with the `mnist_model.h5` file; the confusion matrix serves as a visual representation of the accuracy of the model.

## License
[MIT](https://choosealicense.com/licenses/mit/)