## Larger CNN for the MNIST Dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from emnist import list_datasets
import tensorflow as tf
list_datasets()
['balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist']
from emnist import extract_training_samples
images_train, labels_train = extract_training_samples('letters')
images_train.shape
(240000, 28, 28)
labels_train.shape
(240000,)

from emnist import extract_test_samples
images_test, labels_test = extract_test_samples('letters')
images_test.shape
(40000, 28, 28)
labels_test.shape
(40000,)

images_train = images_train.reshape((images_train.shape[0], 28, 28, 1)).astype('float32')
images_test = images_test.reshape((images_test.shape[0], 28, 28, 1)).astype('float32')
# normalize inputs from 0-255 to 0-1
images_train = images_train / 255
images_test = images_test / 255
# one hot encode outputs
labels_train = np_utils.to_categorical(labels_train)
labels_test = np_utils.to_categorical(labels_test)
num_classes = labels_test.shape[1]

model = tf.keras.models.load_model('model.h5')

scores = model.evaluate(images_test, labels_test, verbose=0)

print("Error: %.2f%%" % (100-scores[1]*100))
print("Accuracy: %.2f%%" % (scores[1]*100))