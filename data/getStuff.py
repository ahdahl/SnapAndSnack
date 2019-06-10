import sys
import csv
import re
import json
from multiprocessing.pool import ThreadPool
import requests
import shutil
pics_csv = sys.argv[1]
# names_csv = sys.argv[2]

MIN_PICS_PER_FOOD = 100
MAX_PICS_PER_FOOD = 125
IMG_SAVE_PATH = 'imgs100/'

with open(pics_csv) as f:
    picslist = f.readlines()
    picslist = [x.strip() for x in picslist]

# with open(names_csv) as  f:
#     nameslist = f.readlines()
#     nameslist = [x.strip() for x in nameslist]

valid = []
for i, pic in enumerate(picslist):
    if pic != "" and pic !=  "error" and len(pic.split(",")) >= MIN_PICS_PER_FOOD and len(pic.split(",")) <= MAX_PICS_PER_FOOD:
            valid.append(pic.strip())
# print(len(valid))
valid = valid[:200]
# print(len(valid))
# recipes = {}
# for i, (nameandrecipe, pic) in enumerate(valid):
#     splitted = nameandrecipe.split(",\"")
#     name = splitted[0].split("/")[-1][:-5]
#     recipe = splitted[1].split("\"")[0]
#     ingredients = re.findall(r"@(.*?),", recipe)
#     recipes[name] = ingredients
#
# with open('recipes.json', 'w') as json_file:
#   json.dump(recipes, json_file)

def f(tupples):
    path, url = tupples
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in r:
                f.write(chunk)
tupples = []
for i, pic in enumerate(valid):
    urls = pic.split(",")
    for i, url in enumerate(urls):
        if url != "":
            splitted = url.split("420x280-fix-")
            name = splitted[1]
            name = name[:-4]
            tupples.append((IMG_SAVE_PATH  + name + str(i)+".jpg",url))

p = ThreadPool(100)
p.map(f, tupples)
