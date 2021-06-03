import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from lib.utils import kl_divergence, category_2_one_hot

data_name = 'cifar10'
model_name = 'ResNet18'
opt = 'adam'
lr = '0.0001'
data_augmentation = True
noise_split = 0.1
runs = 'runs/aug_{}_noise_{}_opt_{}_lr_{}'.format(data_augmentation, noise_split, opt, lr),\
       data_name, "{}".format(model_name)
loss_type = "zo"
data_dir = ''

root_dir = os.path.join(runs, data_name, model_name.lower())
test_id_list = range(0, 5, 1)
max_epoch = 250

data_root = os.path.join(data_dir, data_name)
if data_name == 'cifar10':
    dataset = datasets.CIFAR10
    nb_class = 10
elif data_name == 'cifar100':
    dataset = datasets.CIFAR100
    nb_class = 100
elif data_name == 'svhn':
    dataset = datasets.SVHN
    nb_class = 10
else:
    raise Exception('No such dataset')

eval_transform = transforms.Compose([transforms.ToTensor()])
if 'cifar' in data_name:
    test_data = dataset(data_root, train=False, transform=eval_transform)
    targets = np.array(test_data.targets)
elif 'svhn' in data_name:
    test_data = dataset(data_root, split='test', transform=eval_transform)
    targets = np.array(test_data.labels)

targets_one_hot = np.zeros(shape=(len(targets), nb_class))
targets_one_hot[np.arange(len(targets)), targets] = 1.

var_list = []
bias_list = []
risk_list = []

for epoch in range(max_epoch):
    probs_list = []
    for test_id in test_id_list:
        probs = np.load(os.path.join(root_dir, '{}'.format(test_id), '{}.npz'.format(epoch+1)))['probs']
        probs_list.append(probs)

    probs_list = np.array(probs_list)

    # MSE
    if loss_type == 'mse':
        mean = np.mean(probs_list, axis=0)
        bias = np.mean((targets_one_hot-mean)**2)
        var = np.mean(np.std(probs_list, axis=0)**2)
        risk = np.mean((probs_list-targets_one_hot[np.newaxis, :, :])**2)

    # Zero-one Loss
    elif loss_type == 'zo':
        label_list = np.argmax(probs_list, axis=-1)
        probs_list = np.array([category_2_one_hot(_, nb_class) for _ in label_list])
        mean = np.argmax(np.mean(probs_list, axis=0), axis=-1)
        # mean = category_2_one_hot(mean, nb_class)
        bias = np.mean(np.argmax(targets_one_hot, axis=-1) != mean)
        var = np.mean([_ != mean for _ in label_list])
        risk = 1 - nb_class * np.mean((probs_list*targets_one_hot[np.newaxis, :, :]))

    # CE
    elif loss_type == 'ce':
        mean = np.exp(np.mean(np.log(probs_list+1e-10), axis=0, keepdims=True))
        mean = mean/np.sum(mean, axis=-1, keepdims=True)
        bias = np.mean(kl_divergence(targets_one_hot, mean))
        var = np.mean(kl_divergence(mean[np.newaxis, :, :], probs_list))
        risk = np.mean(kl_divergence(targets_one_hot[np.newaxis, :, :], probs_list))

    var_list.append(var)
    bias_list.append(bias)
    risk_list.append(risk)

var_list = np.array(var_list)
bias_list = np.array(bias_list)
risk_list = np.array(risk_list)
# np.savez('figuretemp/{}_{}_{}_{}.npz'.format(data_name, model_name, opt, lr),
#          loss=risk_list, bias=bias_list, var=var_list)
fig = plt.figure(figsize=(3.5, 3))
ax = fig.add_subplot(111)
ax.set_xlabel('epoch', fontsize=14)
ax.set_xlim([1, max_epoch])

ax.set_ylim([0.05, 0.3])

ax.set_ylabel('loss/bias/var', fontsize=14)
ax.plot(range(1, max_epoch+1, 1), risk_list, label='loss')
ax.plot(range(1, max_epoch+1, 1), bias_list, label='bias')
ax.plot(range(1, max_epoch+1, 1), var_list, label='var')

# ax.legend(fontsize=14, loc='upper right')

# ax.set_xscale('log')
# ax.set_yscale('log')

# plt.title('{} (fail)'.format(model_name), fontsize=14)
plt.title('{}'.format(model_name), fontsize=14)
plt.tight_layout()
plt.show()