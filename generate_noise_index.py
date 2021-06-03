import numpy as np
import scipy.io as sio
import random


# data_name = 'svhn'
data_name = 'cifar'

if 'cifar' in data_name:
    data_size = 50000
elif 'svhn' in data_name:
    data_size = 73257
# random_index_list = []
index_dict = {}
for noise_split in np.arange(0.05, 0.41, 0.05).round(2):
    print(noise_split)
    random_index = random.sample(range(data_size), int(data_size*noise_split))
    # random_index_list.append(random_index)
    index_dict['n{}'.format(int(noise_split*100))] = random_index

# np.savez('{}_noise_record.npz'.format(data_name), n1=random_index_list[0], n2=random_index_list[1])
# sio.savemat('{}_noise_record.mat'.format(data_name), index_dict)
