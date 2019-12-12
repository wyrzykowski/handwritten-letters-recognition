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
import cv2
from emnist import list_datasets

#ładowanie danych do nauki i testowania
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

# w Keras'ie warstwy uzywane do 2 wymiarowego zwieniecia( convolutions ) sa w formie:  [pixels][width][height][channels]
#(images_train.shape[0] - to jest ilosc wierszy w macierzy(czyli ilosc probek)
# zjdecie ma rozmiae 28 x 28 - to siedzi w macierzy - robie z tego tablice jedno wymiarowa o rozmiarze 28x28=784
#to float32 ustawia precyzje pixela na 32 bity
# reshape to be [samples][width][height][channels] chanles =1 bo skala szarosci
images_train = images_train.reshape((images_train.shape[0], 28, 28, 1)).astype('float32') #chyba ta jedynka robi to, że jest jeden wymiar, czyli skala szarosci a nie kolorowo
images_test = images_test.reshape((images_test.shape[0], 28, 28, 1)).astype('float32')

#pixele sa teraz w skali szarosci od 0 do 255
# normalize inputs from 0-255 to 0-1
images_train = images_train / 255
images_test = images_test / 255

# transforming the vector of class integers into a binary matrix.
# gdzie class oznacza jeden label np litere A
labels_train = np_utils.to_categorical(labels_train)
labels_test = np_utils.to_categorical(labels_test)

num_classes = labels_test.shape[1] # ilosc liter jaka bedzie rozpoznawana = 26 od 1 do 27 - takie maja labele

# define the larger model
def larger_model():
	# create model
	model = Sequential()
	#Conv2D - tutaj jako warstwa wejsciowa
	model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu')) # 30 - liczba wyjsciowych filtrów, rozmiar filtra 5 x 5
	# krok defaultowo = (1,1)
	# funkcja relu jest stosowana do wyjscia z pojedynczych filtrów

	model.add(MaxPooling2D()) #jak nic nie podam to zmniejszy rozmiar dwukrotnie
	# warstwa MaxPooling2D rozwiazuje problem warstw spolotowych, które są czułe na lokalozacje wystepowania cechy na obrazie
	# rozwiązaniem jest próbkowanie w dół - te warstwy rozwiązaują to poprzez sumowanie wystepowania cech w mapy płatów
	# 2D bo chyba dwa wymiary ma zdj, Max - bo tutaj sumowane jest tylko najbardziej widoczna obecność cechy (istnieje tez Average)
	model.add(Conv2D(16, (3, 3), activation='relu'))
	model.add(MaxPooling2D()) # defaultowo ma filtr 2x2
	model.add(Dropout(0.2)) # losowo wyklucza 20% neuronów w warstwie, aby zmniejszyć nadmierne dopasowanie.
	model.add(Flatten()) # splaszcza dane z wielowymairowej tablicy do jednowymairowej tak by byla mozliwosc
	# przeslania informacji pojedynczo do kolejnej warstwt a nie wsadzanie tablicy do pojedynczego neuronu
	model.add(Dense(128, activation='relu')) #warstwa, ktora laczy wszystkie wezly z jedej warstwy dodrugiej (kazdy do kazdego)
	# 128 to ilosc wezlow w tej warstwie, relu - funkcja aktywacji
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax')) # funkcja aktywacji softmax robi z z danych wyjsciowych wartosci prawdopodobienstwa
	# ogolnie funkcje aktywacji maja za zadanie skurczyc dane
	# i pozwala wybrac jedna klase z 26 jako prawdopodbny wynik modelu
	# po prostu ustawia wartosci na kazdym neuroanie, że wszystkie sumuja sie do 1


	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# loss - mowi jak bardzo sie pomylil model,
	# otimizer -mowi w jaki sposb sie doiwemy ze idzie lepiej
	# metrics - co jest wytyczna - precyzyjnosc
	return model
# build the model
model = larger_model()

# Fit the model - Train the model
model.fit(images_train, labels_train, validation_data=(images_test, labels_test), epochs=25, batch_size=200)
# batch_size - mówi po ilu przepatrzonych zdjeciach model sie updatuje
# epochos ile razy przeleci to samo

# Final evaluation of the model
scores = model.evaluate(images_test, labels_test, verbose=0)


#save the model to disk
model.save('model.h5')

#classification error rate
print("Error: %.2f%%" % (100-scores[1]*100))
print("Accuracy: %.2f%%" % (scores[1]*100))

# loss function - to sposob obliczania bledu - są różne ich rodzaje, ogolnie mówia jak zła odpowiedz jest
# ogólnie: porównują one wynik z tym co powinno wyjść - zwracają wynik i na tje podstawie dostosowuje bias(przekłamanie) i wagi

# ReLu - funkcja aktywacji (Restified linear Unit) - dla wartosci ujemnej daje 0, dla do datnich po prostu je zwraca

# sigmoid function - funkcja aktywacji mapuje kazda dana, ktora sie jej da na wartosc pomiedzy -1 a 1
# to jest funkcja nieliniowa,robi tak, że im wartosc dana jest blizej nieksonczonosci to zwraca wynik blziszy do 1
# a jesli wynik blizszy jest -nieskonczonosci to zwraca wartosc blizsza -1
# czyli dodaje zlonoosci, bo to juz funkcja nieliniowa jest


# Warstwy splotowe (Convolutional) podsumowują obecność cech na obrazie wejściowym

# Pooling - Zmniejszenie macierzy (lub macierzy) utworzonej przez wcześniejszą warstwę splotową do mniejszej macierzy.
# Operacja łączenia, podobnie jak operacja splotowa, dzieli tę macierz na plasterki,
# a następnie przesuwa tę operację splotową o kolejne kroki.

# ideą Convolutions( spolotów ) jest patrzenie nie na caly obrazek ale na jakas jego czesc (patch - plaster)
# bedziemy używac filtra, który jest dużo mniejszy od obrazka

# convolutional operation - 2 korki matematycznych operacji:
# 1.  mnożenie filtra splotowego i wycinka macierzy wejściowej. (Wycinek macierzy wejściowej ma tę samą rangę i rozmiar co filtr splotowy.)
# Sumowanie wszystkich wartości w wynikowej macierzy produktu.
# filtry splotowe są zwykle zapełniane liczbami losowymi, a następnie sieć trenuje wartości idealne.