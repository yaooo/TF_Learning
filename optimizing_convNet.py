import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

# NAME = "Cats-vs-Dogs-cnn-64x2-{}".format(int(time.time()))


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.333)
sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0


import time

dense_layers = [0,1,2]
layer_sizes = [32,64,128]
conv_layers = [1,2,3]

for dense_layer in dense_layers:
	for layer_size in layer_sizes:
		for conv_layer in conv_layers:
			NAME = "{}-cov-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
			print(NAME)
			tensorborad = TensorBoard(log_dir='logs/{}'.format(NAME))

			model = Sequential()

			model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2, 2)))

			for l in range(conv_layer-1):
				model.add(Conv2D(64, (3, 3)))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2, 2)))

			model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

			for l in range(dense_layer):
				model.add(Dense(layer_size))
				mode.add(Activation('relu'))
			# model.add(Dense(64))
			# model.add(Activation('relu'))

			model.add(Dense(1))
			model.add(Activation('sigmoid'))

			model.compile(loss='binary_crossentropy',
			              optimizer='adam',
			              metrics=['accuracy'])

			model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorborad])

##########################################################################################
# Command for calling tensorboard: 
#tensorboard --logdir=data/ --host localhost --port 8088
##########################################################################################