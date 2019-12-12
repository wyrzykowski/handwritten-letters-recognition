from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


# Larger CNN for the MNIST Dataset
# from emnist import list_datasets
# from mnist import MNIST
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

from emnist import list_datasets
list_datasets()['balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist']
from emnist import extract_training_samples
images, labels = extract_training_samples('digits')
images.shape(240000, 28, 28)
labels.shape(240000,)

# load data
(X_train, y_train), (X_test, y_test) = emnist.load_data()
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define the larger model

labels = [0, 1,2, 3, 4, 5, 6, 7, 8, 9]

def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = larger_model()

model.load_weights('model.h5')

test_image = cv2.imread('./test2.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(test_image)
test_image = cv2.resize(test_image,(28,28))
plt.show()

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

# print(y_test[int(model.predict(test_image))])

predictions = model.predict(test_image)

indexs = np.argmax(predictions, axis =1)

print("Recognized: ", labels[indexs[0]])