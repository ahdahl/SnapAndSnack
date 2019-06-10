import os
import numpy as np
from matplotlib import pyplot as plt

DIR_FIXED = 'imgs_fixed_100'
arr = []

for r, d, files in os.walk(DIR_FIXED):
	# arr.append((r[11:], len(files)))
	arr.append(len(files))
print(arr)

cpt = sum([len(files) for r, d, files in os.walk(DIR_FIXED)])
print(cpt)
arr = arr[1:]

arr = sorted(arr)[6:]
print(arr)
# fixed bin size
# bins = np.arange(-100, 100, 5) # fixed bin size

plt.xlim([min(arr)-5, max(arr)+5])

plt.hist(arr, alpha=0.5)
plt.title('Class Imbalance')
plt.xlabel('Number of Images')
plt.ylabel('Number of Classes')
plt.savefig("ClassImbalanceNoBarbie.png")
plt.show()
