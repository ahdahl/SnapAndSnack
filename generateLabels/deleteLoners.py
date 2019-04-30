import os
delete = []
from os.path import join
path = os.path.dirname(os.path.abspath(__file__)) + '/smooth'
print(path)
for root, dirs, files in os.walk('smooth'):
    files = sorted(files)
    for i in range(1, len(files) - 1):
        file = files[i]
        # if file.endswith('.JPEG'):
        #     os.rename(join(path,file), join(path,file[:-5]) + '.jpg')
        if file.endswith('.jpg'):
            nextfile = files[i + 1]
            if nextfile[:-4] != file[:-4]:
                delete.append(file)
        if file.endswith('.txt'):
            prevfile = files[i - 1]
            if prevfile[:-4] != file[:-4]:
                delete.append(file)
# for file in delete:
#     os.remove('smooth/' + file)
print(delete)
