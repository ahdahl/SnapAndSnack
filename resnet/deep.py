from time import time
import matplotlib.pyplot as plt
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from keras import optimizers
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import sys

# fix random seed for reproducibility
seed = 100

# input size 1000 for ResNet50
input = 6096
numpy.random.seed(seed)
data_name = sys.argv[1]

# load dataset
dataframe = pandas.read_csv(data_name, header=None)
dataset = dataframe.values
X = dataset[:,0:input].astype(float)
Y = dataset[:,input]

# hot encoding
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	# model.add(Dense(2048,input_dim=input, activation='relu'))
	# model.add(Dropout(0.4))
	# model.add(Dense(128, activation='relu'))
	# model.add(Dropout(0.2))
	# model.add(Dense(32, activation='relu'))
	# model.add(Dropout(0.2))
	# model.add(Dense(8, activation='relu'))
	model.add(Dense(390, activation='softmax'))
	# adam = optimizers.Adam(.001)
	model.compile(loss='categorical_crossentropy',optimizer = 'adam' , metrics=['accuracy'])
	return model

def get_activations(model):
	model2 = Sequential()
	model2.add(Dense(512, input_dim=input, activation='relu', weights=model.layers[0].get_weights()))
	model2.add(Dense(128, activation='relu', weights=model.layers[2].get_weights()))
	model2.add(Dense(32, activation='relu', weights=model.layers[4].get_weights()))
	model2.add(Dense(8, activation='relu', weights=model.layers[6].get_weights()))
	activations = model2.predict(X)
	return activations

def save_model(model):
    # saving model
    json_model = model.to_json()
    open('model_architecture.json', 'w').write(json_model)
    # saving weights
    model.save_weights('model_weights.h5', overwrite=True)

def load_model():
    # loading model
    model = model_from_json(open('model_architecture.json').read())
    model.load_weights('model_weights.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

model = baseline_model()
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(X_train, Y_train, nb_epoch=40, batch_size=64, verbose=1, callbacks= [tensorboard])

#model = load_model()

scores = model.evaluate(X_test, Y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# activations = get_activations(model)
#
# #make plot from activations
# dic = {}
# for i in range(len(Y)):
# 	label = Y[i]
# 	if label not in dic:
# 		dic[label] = [[activations[i][0]],[activations[i][1]]]
# 	else:
# 		dic[label][0] += [activations[i][0]]
# 		dic[label][1] += [activations[i][1]]
# plt.scatter(dic['banana'][0],dic['banana'][1],color='yellow', alpha = .1)
# plt.scatter(dic['spinach'][0],dic['spinach'][1],color='green', alpha = .1)
# plt.scatter(dic['strawberry'][0],dic['strawberry'][1],color='red', alpha = .1)
# # plt.show()
# plt.savefig('plot.png',dpi=300)

save_model(model)
