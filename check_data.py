
## This file will check if you have the required data. If not, it will automatically download it for you.


# Image Classification
# In this project, I classify images from the CIFAR-10 dataset(https://www.cs.toronto.edu/~kriz/cifar.html).  The dataset consists of airplanes, dogs, cats, and other objects.

# Get the Data
# Run the following cell to download the CIFAR-10 dataset for python(https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).


from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm

import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

# check the if the data is ready
floyd_cifar10_location = '/cifar/cifar-10-python.tar.gz'
if isfile(floyd_cifar10_location):
    tar_gz_path = floyd_cifar10_location
else:
    tar_gz_path = 'cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_path,
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open(tar_gz_path) as tar:
        tar.extractall()
        tar.close()