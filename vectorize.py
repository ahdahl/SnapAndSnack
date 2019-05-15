
import numpy as np


import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input as preprocess_input_vgg
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
from keras.models import Model, load_model


model = keras.applications.VGG16(weights='imagenet', include_top=True)
# # get_image will return a handle to the image itself, and a numpy array of its pixels to input the network
def get_image_vgg(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_vgg(x)
    return img, x


# import sys

feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

# np.savetxt("here.txt",feat)


import os
path = os.path.dirname(os.path.abspath(__file__)) + '/generateLabels/smooth'
# print(path)
for root, dirs, files in os.walk('generateLabels/smooth'):
	files = sorted(files)
	for i in range(len(files)):
		file = files[i]
		if file.endswith('.jpg'):
			img, x = get_image_vgg("generateLabels/smooth/" + file)
			feat = feat_extractor.predict(x)
			print(feat.shape)
			# print("saved: " + file[:-4])
			# np.save("vectors/" + file[:-4], feat)


# from keras import metrics
# import pandas as pd
# import _pickle as pickle
# from pprint import pprint
# import random
# #from scipy.spatial import distance
# #from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# from keras.utils.np_utils import to_categorical
# from os import listdir
# from os.path import isfile, join
# import shutil
# import stat
# import collections
# from collections import defaultdict

# from google.colab import files
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials
# import psutil
# from tensorflow.python.client import device_lib

# import tables
# import falconn
#from annoy import AnnoyIndex

  
# def get_image_inc(path):
#     img = image.load_img(path, target_size=(299, 299))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input_inc(x)
#     return img, x
  
# def plot_preds(image, preds, top_n):  
#     plt.imshow(image)
#     plt.axis('off')
#     plt.figure()
    
#     order = list(reversed(range(top_n)))
#     labels = [categories[x] for i, x in enumerate(np.argsort(-probabilities)[:top_n])]
#     bar_preds = [-np.sort(-probabilities)[i] for i, x in enumerate(np.argsort(-probabilities)[:top_n])]
    
#     plt.barh(order, bar_preds, alpha=0.8, color='g')
#     plt.yticks(order, labels, color='g')
#     plt.xlabel('Probability', color='g')
#     plt.xlim(0, 1.01)
#     plt.tight_layout()
#     plt.show()

# def show_result_images(final_result):
#     rows = 2
#     cols = 3
#     fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(12, 12))
#     fig.suptitle('Result Images from Query', fontsize=20)
#     food_dirs = [food_direction[4] for food_direction in final_result]
#     for i in range(rows):
#       for j in range(cols):
#         food_dir = food_dirs[i*cols + j]
#         img = plt.imread(food_dir)
#         ax[i][j].imshow(img)
#         ec = (0, .6, .1)
#         fc = (1, 1, 1)
#         ax[i][j].text(0, 0, get_corresponding_recipes(final_result).recipe_name[i*cols + j], size=15, rotation=0,
#                 ha="left", va="top", 
#                 bbox=dict(boxstyle="round", ec=ec, fc=fc))
#     plt.setp(ax, xticks=[], yticks=[])
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# matplotlib.pyplot.figure(figsize=(16,4))
# matplotlib.pyplot.plot(feat[0])
# matplotlib.pyplot.show()
