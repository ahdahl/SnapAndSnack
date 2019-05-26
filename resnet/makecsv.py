import os
import numpy as np
import csv
# print(path)
import sys
vecfolder = sys.argv[1]
output =sys.argv[2]
path = os.path.dirname(os.path.abspath(__file__)) + '/' + vecfolder

out = []
for root, dirs, files in os.walk(vecfolder):
    files = sorted(files)
    for i in range(len(files)):
        file = files[i]
        if file.endswith(".npy"):
            vec = np.load(vecfolder + "/" + file)[0]
            file = file[:-4]
            remove = 0
            for i in range(len(file) - 1,0,-1):
                if file[i].isdigit():
                    remove += 1
                else:
                    break
            file = file[:-remove]
            vecstr = ','.join([str(e) for e in vec])
            out.append([vecstr + ',' + file])

with open(output, 'w') as myfile:
     wr = csv.writer(myfile, delimiter = "\n")
     for line in out:
         wr.writerow(line)


# with open(classes_path, 'r') as f:
#     reader = csv.reader(f)
#     classes = list(reader)
# out = []
# for root, dirs, files in os.walk(vecfolder):
#     files = sorted(files)
#     for i in range(len(files)):
#         file = files[i]
#         if file.endswith(".npy"):
#             vec = np.load(vecfolder + "/" + file)[0]
#             vecstr = ','.join([str(e) for e in vec])
#             for j in classes:
#                 if j[0] in file:
#                     out.append([vecstr + ',' + j[1]])
#
# with open(output, 'w') as myfile:
#      wr = csv.writer(myfile, delimiter = "\n")
#      for line in out:
#          wr.writerow(line)
