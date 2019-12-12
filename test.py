import cv2
from tensorflow.keras.preprocessing import image
# Larger CNN for the MNIST Dataset
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

# define the larger model
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
# build the model
model = larger_model()

labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

model.load_weights('model.h5')

test_image = cv2.imread('./test-u.png', cv2.IMREAD_GRAYSCALE)
test_image = ~test_image;
plt.imshow(test_image)
test_image = cv2.resize(test_image,(28,28))
plt.show()

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

predictions = model.predict(test_image)

index = np.argmax(predictions, axis = 1)

print("Recognized: ", labels[index[0]-1])