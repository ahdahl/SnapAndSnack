import os
import numpy as np
import csv
import sys
# vgg = sys.argv[1]
# resnet = sys.argv[2]
# incep = sys.argv[3]
# output = sys.argv[4]

vgg = sys.argv[1]
resnet = sys.argv[2]
# incep = sys.argv[3]
output = sys.argv[3]
# path = os.path.dirname(os.path.abspath(__file__)) + '/' + vgg

out = []
for root, dirs, files in os.walk(vgg):
    files = sorted(files)
    for i in range(len(files)):
        file = files[i]
        # print(file)
        if file.endswith(".npy"):
            vecvgg = np.load(vgg + "/" + file)[0]
            vecres = np.load(resnet + "/" + file)[0]
            # vecinc = np.load(incep + "/" + file)[0]
            # print(vecres)
            file = file[:-4]
            remove = 0
            for i in range(len(file) - 1,0,-1):
                if file[i].isdigit():
                    remove += 1
                else:
                    break
            file = file[:-remove]
            vecstrvgg = ','.join([str(e) for e in vecvgg])
            vecstrres = ','.join([str(e) for e in vecres])
            # vecstrinc = ','.join([str(e) for e in vecinc])
            # out.append([vecstrvgg + ',' + file])
            out.append([vecstrvgg + ',' + vecstrres + ',' + file])
            # out.append([vecstrvgg + ',' + vecstrres + ',' + vecstrinc + ',' + file])

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
