### About the data set

#### Image Classification
In this project, I classify images from the CIFAR-10 dataset(https://www.cs.toronto.edu/~kriz/cifar.html).  The dataset consists of airplanes, dogs, cats, and other objects.

#### Get the Data
Run the following cell to download the CIFAR-10 dataset for python(https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

### The python files

In order to keep the size of this repository as small as possible, the data/images used in this project will be downloaded in the code.

* chekc_data.py   ---- Check if the required data is in the right place
* preprocessing.py   ---- Do the data-preprocessing, including one-hot-encoding, normalization, train-test splitting.
* build_network.py   ---- Steps to build a convolutional neural network
* set_params.py   ---- Set up parameters for training the network
* training.py   ---- Train the classifier
* test.py   ---- Test the classifier
* image_classification.ipynb   ---- Jupyter notebook version, a better version to showcase the results


### The classification results

#### Here is the traning accuracy
![Training Accuracy](Results/training_acc.jpg?raw=true "Title")

#### And here is the predictions made by the classifier
![Predictions](Results/predicting.png?raw=true "Title")
