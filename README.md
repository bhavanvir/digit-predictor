# Digit-Prediction

Digit-Prediction is a command-line, machine learning preceptron that uses a neural network to classify handwritten digits. The structure of the neural network consists of an input layer of 784 neurons, a hidden layer of 100 neurons, and an output layer of 10 neurons. 

The [MNIST](https://keras.io/api/datasets/mnist/) digits classification dataset was used for training; the dataset consists of 60,000 28x28 grayscale images of 10 digits, along with a test set of 10,000 images. The neural network is designed using the [SGD](https://keras.io/api/optimizers/sgd/) optimizer and the [categorical cross-entropy](https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class) loss function.

## Installation

Several dependencies are required to run Digit-Prediction that are not included in the Python Standard Library. It is imperative that these modules are installed and functional beforehand.

Using the package manager [pip](https://pip.pypa.io/en/stable/) to install external modules:
```bash
pip install cv2
pip install matplotlib 
pip install tensorflow # Keras is installed along with TensorFlow
pip install colorama
pip install termcolor
pip install seaborn
```

## Usage

To use the built-in Digit-Prediction test harness, have the digit wanting to be predicted successfully in the name of the file. For example, if the image is of digit 3, the file name should be `test_3.jpg` or something similar.

If the `mnist_model.h5` file is not present in the root directory, the Digit-Prediction application will first run a training session to generate the model; the default epoch length is defined as `10000`, but this can be changed by altering the `epochs` parameter within the `new_model` function.
```python
model.fit(x_train, y_train, batch_size=64, epochs=10000, validation_data=(x_test, y_test))
```
Running the Digit-Prediction application will prompt the user with two branching options: `E` to use your own external data, or `M` to use the MNIST data. 

1. If `E` is selected, the user will be prompted with the option to view the processed version of their input image, or to recieve a command-line prediction along with each image read from the `input` directory.
2. If `M` is selected, the user will be prompted with the option to view an incorrectly predicted image, or a correctly predicted image. In either case, the user will then be asked to input a digit between 0 and 9 inclusive, and the application will then show an example.

## License
[MIT](https://choosealicense.com/licenses/mit/)