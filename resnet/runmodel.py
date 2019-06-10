import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input as preprocess_input_vgg
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
from keras.applications.resnet50 import preprocess_input
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

TRAIN_DIR = '../data/imgs_fixed_train_100'
TEST_DIR = '../data/imgs_fixed_test_100'
HEIGHT,WIDTH = 280,420
TRAINABLE_LAYERS = 54
NUM_TRAIN = 15347
NUM_TEST = 5025
NUM_CLASSES = 184
NUM_EPOCHS = 5
BATCH_SIZE = 32

model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=60,
      horizontal_flip=True,
    )

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                target_size=(HEIGHT, WIDTH),
                                                batch_size=BATCH_SIZE)

def build_finetune_model(base_model, num_classes):
    for layer in model.layers[:-TRAINABLE_LAYERS]:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    #
    # New FC layer, random init
    x = Dense(1024, activation='relu')(x)
    x = Dropout(rate = .5)(x)
    x = Dense(1024, activation='relu')(x)
    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)


    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    return finetune_model
finetune_model = build_finetune_model(model, num_classes=NUM_CLASSES)

from keras.optimizers import Adam

#load pretrained model
#finetune_model.load_weights('weights100_2fc_15e_54nt_32b.h5')


adam = Adam(lr=.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

#from keras.utils import plot_model
#plot_model(finetune_model, to_file='resmodel2.png')

tensorboard = TensorBoard(log_dir="logs/{}".format(time()), update_freq='batch')

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8,
                                       shuffle=True,steps_per_epoch = NUM_TRAIN // BATCH_SIZE,
                                       callbacks=[tensorboard], verbose = 1)
#save model
#finetune_model.save_weights('weights100_2fc_20e_54nt_32b.h5')

datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input
    )

generator = datagen.flow_from_directory(
        TEST_DIR,
        target_size=(HEIGHT, WIDTH),
        batch_size=1,
        shuffle=False)
predictions = finetune_model.predict_generator(generator,NUM_TEST,workers = 8, verbose = 1)

val_preds = np.argmax(predictions, axis=-1)
val_trues = generator.classes
labels = generator.class_indices
cr = classification_report(val_trues, val_preds, target_names=labels)
print(cr)
acc =  accuracy_score(val_trues, val_preds)
print(acc)
