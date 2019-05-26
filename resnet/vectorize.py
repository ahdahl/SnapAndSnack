import numpy as np
import keras
from keras.preprocessing import image
# from keras.applications.imagenet_utils import preprocess_input as preprocess_input_vgg
from keras.applications.resnet50 import preprocess_input
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50

model = ResNet50(weights='imagenet', include_top=True)
# # get_image will return a handle to the image itself, and a numpy array of its pixels to input the network

def get_image_res(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x
from keras.utils import plot_model
plot_model(model, to_file='model.png')

feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc1000").output)

import os

def writeThings(pathname):
    path = os.path.dirname(os.path.abspath(__file__)) + pathname
    count = 0
    for root, dirs, files in os.walk(pathname):
        files = sorted(files)
        for i in range(len(files)):
            file = files[i]
            if file.endswith('.jpg'):
                img, x = get_image_res(pathname + "/" + file)
                feat = feat_extractor.predict(x)
                print("saved: " + str(count) +'/'+ str(len(files)))
                np.save("foodvectorsresnet50/" + file[:-4], feat)
                count += 1
import sys
pathname = sys.argv[1]
writeThings(pathname)
