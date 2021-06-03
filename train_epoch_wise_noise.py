from lib.ModelWrapper import ModelWrapper
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms, datasets
import numpy as np
import random
import sys
import os

args = sys.argv

data_name = args[1]     # 'svhn', 'cifar10', 'cifar100'
model_name = args[2]    # 'resnet18', 'resnet34', 'vgg16', 'vgg13', 'vgg11'
noise_split = float(args[3])
data_augmentation = bool(args[4])
opt = args[5]
lr = float(args[6])
test_id = int(args[7])
data_dir = '#input your data root dir#'

# setting
data_split = 0.5
train_batch_size = 128
train_epoch = 250
eval_batch_size = 250

data_root = os.path.join(data_dir, data_name)
if data_name == 'cifar10':
    dataset = datasets.CIFAR10
    from archs.cifar10 import vgg, resnet
elif data_name == 'cifar100':
    dataset = datasets.CIFAR100
    from archs.cifar100 import vgg, resnet
elif data_name == 'svhn':
    dataset = datasets.SVHN
    from archs.svhn import vgg, resnet
else:
    raise Exception('No such dataset')

if model_name == 'vgg11':
    model = vgg.vgg11_bn()
elif model_name == 'vgg13':
    model = vgg.vgg13_bn()
elif model_name == 'vgg16':
    model = vgg.vgg16_bn()
elif model_name == 'resnet18':
    model = resnet.resnet18()
elif model_name == 'resnet34':
    model = resnet.resnet34()
else:
    raise Exception("No such model!")

if data_augmentation:
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
else:
    train_transform = transforms.Compose([transforms.ToTensor()])
eval_transform = transforms.Compose([transforms.ToTensor()])

# load data
if 'cifar' in data_name:
    train_data = dataset(data_root, train=True, transform=train_transform, download=False)
    if noise_split != 0:
        train_targets = np.array(train_data.targets)
        noise_index = np.load('cifar_noise_record.npz')['n{}'.format(int(10*noise_split))]
        random_part = train_targets[noise_index]
        np.random.shuffle(random_part)
        train_targets[noise_index] = random_part
        train_data.targets = train_targets.tolist()

    train_targets = np.array(train_data.targets)
    train_x = np.array(train_data.data)
    data_size = len(train_targets)
    sampled_size = int(data_size*data_split)
    random_index = random.sample(range(data_size), sampled_size)
    train_data.data = train_x[random_index]
    train_data.targets = train_targets[random_index].tolist()

    test_data = dataset(data_root, train=False, transform=eval_transform)

elif 'svhn' in data_name:
    train_data = dataset(data_root, split='train', transform=train_transform, download=True)

    if noise_split != 0:
        train_targets = np.array(train_data.labels)
        noise_index = np.load('svhn_noise_record.npz')['n{}'.format(int(10*noise_split))]
        random_part = train_targets[noise_index]
        np.random.shuffle(random_part)
        train_targets[noise_index] = random_part
        train_data.labels = train_targets.tolist()

    train_targets = np.array(train_data.labels)
    train_x = np.array(train_data.data)
    data_size = len(train_targets)
    sampled_size = int(data_size*data_split)
    random_index = random.sample(range(data_size), sampled_size)
    train_data.data = train_x[random_index]
    train_data.labels = train_targets[random_index].tolist()

    test_data = dataset(data_root, split='test', transform=eval_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0,
                                           drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=0,
                                          drop_last=False)

# build model
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

if opt == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif opt == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

wrapper = ModelWrapper(model, optimizer, criterion, device)

# train the model
save_path = os.path.join('runs', 'aug_{}_noise_{}_opt_{}_lr_{}'.
                         format(data_augmentation, noise_split, opt, lr),
                         data_name, "{}".format(model_name), "{}".format(test_id))

if not os.path.exists(save_path):
    os.makedirs(save_path)
writer = SummaryWriter(log_dir=os.path.join(save_path, "log"), flush_secs=30)

wrapper.train()
for id_epoch in range(train_epoch):
    # train loop
    train_loss = 0
    train_acc = 0
    train_size = 0

    for id_batch, (inputs, targets) in enumerate(train_loader):
        loss, acc, correct = wrapper.train_on_batch(inputs, targets)
        train_loss += loss
        train_acc += correct
        train_size += len(targets)
        print("epoch:{}/{}, batch:{}/{}, loss={}, acc={}".
              format(id_epoch+1, train_epoch, id_batch+1, len(train_loader), loss, acc))
    train_loss /= id_batch
    train_acc /= train_size
    writer.add_scalar("train acc", train_acc, id_epoch+1)
    writer.add_scalar("train loss", train_loss, id_epoch+1)

    # eval
    wrapper.eval()
    probs, _, _ = wrapper.predict_all(test_loader)
    test_loss, test_acc = wrapper.eval_all(test_loader)
    print("epoch:{}/{}, batch:{}/{}, testing...".format(id_epoch + 1, train_epoch, id_batch + 1, len(train_loader)))
    print("clean: loss={}, acc={}".format(test_loss, test_acc))
    writer.add_scalar("test acc", test_acc, id_epoch+1)
    writer.add_scalar("test loss", test_loss, id_epoch+1)
    np.savez(os.path.join(save_path, "{}.npz".format(id_epoch+1)), probs=probs)
    wrapper.train()
writer.close()
