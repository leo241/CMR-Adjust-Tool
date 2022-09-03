import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from SimpleITK import GetArrayFromImage, ReadImage
from PIL import Image
from torch.autograd import Variable
import numpy as np
import warnings

warnings.filterwarnings("ignore")  #ignore warnings
fig_class_list = ['C0','T2','LGE']

def MyLoader(pic_path):
    '''
    用于对图片进行预处理
    :param pic_path: 图片的路径
    :return:
    '''
    img =  GetArrayFromImage(ReadImage(pic_path))
    if len(img.shape) == 3: # 如果是三维图像
        img = img[:,:,0] # 取第一个slice图像作为目标
    img_st = (img - img.mean())/ img.std() # 将图像像素点进行（0，1）标准化
    return Image.fromarray(img_st) # 要变成Image类型才能后续用transform转换




class MyDataset(Dataset): #
    '''
    继承了torch.utils.data.Dataset,用于加载数据，后续载入神经网络中
    '''
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item): # 这个是Dataset类的关键函数，形成数据的最终形式，通过迭代的形式喂给后续的神经网络
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

def tag2tensor(tag):
    '''
    根据标签创建张量
    :param tag: '000' '001' '010' '011' '100' '101'  '110' '111'
    :return: one hot tensor data
    '''
    if tag == '000':
        return torch.tensor([1.0,0,0,0,0,0,0,0])
    if tag == '001':
        return torch.tensor([0,1.0,0,0,0,0,0,0])
    if tag == '010':
        return torch.tensor([0,0,1.0,0,0,0,0,0])
    if tag == '011':
        return torch.tensor([0,0,0,1.0,0,0,0,0])
    if tag == '100':
        return torch.tensor([0,0,0,0,1.0,0,0,0])
    if tag == '101':
        return torch.tensor([0,0,0,0,0,1.0,0,0])
    if tag == '110':
        return torch.tensor([0,0,0,0,0,0,1.0,0])
    return torch.tensor([0,0,0,0,0,0,0,1.0]) # tag for 111


def get_data(fig_class):
    '''
    把某种类别的图像划分训练集 测试集 验证集，要求工作路径下存在data_png文件夹
    :param fig_class: 'C0','T2','LGE' 中的某一个
    :return: train_list, valid_list, test_list
    '''
    train_path = f'data_png/train_data_{fig_class}'
    valid_path = f'data_png/valid_data_{fig_class}'
    test_path = f'data_png/test_data_{fig_class}'

    train_list = list()
    valid_list = list()
    test_list = list()

    train_tags = os.listdir(train_path)
    for train_tag in train_tags:
        train_pic_names = os.listdir(f'{train_path}/{train_tag}')
        for train_pic_name in train_pic_names:
            train_pic_path = f'{train_path}/{train_tag}/{train_pic_name}'
            train_list.append([train_pic_path, tag2tensor(train_tag)]) # each item in train_list is [image path, image class tag]

    valid_tags = os.listdir(valid_path)
    for valid_tag in valid_tags:
        valid_pic_names = os.listdir(f'{valid_path}/{valid_tag}')
        for valid_pic_name in valid_pic_names:
            valid_pic_path = f'{valid_path}/{valid_tag}/{valid_pic_name}'
            valid_list.append([valid_pic_path, tag2tensor(valid_tag)])

    test_tags = os.listdir(test_path)
    for test_tag in test_tags:
        test_pic_names = os.listdir(f'{test_path}/{test_tag}')
        for test_pic_name in test_pic_names:
            test_pic_path = f'{test_path}/{test_tag}/{test_pic_name}'
            test_list.append([test_pic_path, tag2tensor(test_tag)])

    return train_list, valid_list, test_list


transform = transforms.Compose([ # transform to figure, for further passing to nn
    transforms.Resize((256, 256)),
    transforms.ToTensor(), # ToTensor会给灰度图像自动增添一个维度
])


class cnn(nn.Module): # construction of netral network
    def __init__(self):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d( # 1 224 224
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2 # if stride = 1 padding = (kernel_size - 1)/2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16,128,128
        )
        # 16 224 224
        self.conv2 = nn.Sequential( # 16,128,128
            nn.Conv2d(16,32,5,1,2), # 32 128 128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 32 64 64
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,5,1,2),# 64 32 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 64 16 16
        )
        self.fc1 = nn.Linear(64*32*32, 64)
        self.out= nn.Linear(64, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        # print(x.size(), '进入全连接层前的维度')
        x = self.relu(self.fc1(x))
        x = self.out(x)
        return x



# hyper parameters
lr = 0.01
batch_size = 10 # how much data given to nn per iteration
EPOCH = 40 # you may decrease epoch, but maybe not too small
Max_iteration = 800 # max iteration, should be just devided by 10
                # , you could improve the performance of model by increasing this
                # 300, 600
C0_iter = 2000
T2_iter = 2000
LGE_iter = 2000

Max_iteration_list = [C0_iter, T2_iter, LGE_iter]


if __name__ == '__main__':
    for pic_type in [0,1,2]: # 遍历C0 T2 LEG
        print(f'\n-----training for model <{fig_class_list[pic_type]}>, waiting for <{Max_iteration_list[pic_type]}> iterations please ...-----')
        train_list, valid_list, test_list = get_data(fig_class_list[pic_type]) # 获取C0的训练集 验证集 和测试集

        train_data = MyDataset(train_list, transform=transform, loader=MyLoader)
        valid_data = MyDataset(valid_list, transform=transform, loader=MyLoader)
        test_data = MyDataset(test_list, transform=transform, loader=MyLoader)

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0) # batch_size是从这里的DataLoader传递进去的
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0)

        one_pic = MyLoader(train_list[0][0])
        one_pic_tran = transform(one_pic)

        net = cnn()
        optimizer = torch.optim.Adam(net.parameters(),lr=lr)
        loss_func = nn.BCEWithLogitsLoss()


        # training
        i = 0
        loss_train_list = list()
        loss_valid_list = list()
        iter_list = list()
        stop = False
        for epoch in range(EPOCH):
            if stop == True:
                break
            for step,(x,y) in enumerate(train_loader):
                b_x = Variable(x) # batch x
                b_y = Variable(y) # batch y
                output = net(b_x)
                loss = loss_func(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1


                if i %10 == 0:
                    print(f'iteration: {i}')
                    if i == Max_iteration_list[pic_type]:
                        stop = True
                        torch.save(net, f'model{fig_class_list[pic_type]}_{i}.pkl')
                        break
                    for data in valid_loader:
                        x_valid, y_valid = data
                        output = net(x_valid)
                        valid_loss = loss_func(output, y_valid)
                        loss_train_list.append(float(loss)) # 每隔10个iter，记录一下当前train loss
                        loss_valid_list.append(float(valid_loss)) # 每隔10个iter，记录一下当前valid loss
                        iter_list.append(i) # 记录当前的迭代次数
                        print('train_loss:', float(loss))
                        print('-----valid_loss-----:', float(valid_loss))
                        break
        plt.plot(iter_list,loss_train_list,label = "train loss")
        plt.plot(iter_list,loss_valid_list,label = "valid loss")
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend() # 显示图例文字
        plt.title(f'{fig_class_list[pic_type]} model train and valid loss with iteration')
        plt.savefig(f'model {fig_class_list[pic_type]} loss {i} iter.png')
        print(f'-----{fig_class_list[pic_type]} loss figure saved!-----')
        plt.close()
        torch.save(net, f'model{fig_class_list[pic_type]}_{i}.pkl')
        print(f'-----{fig_class_list[pic_type]} model saved!-----')


print('\n\n ----------all models saved successfully!----------')
# a method for calculate accuracy on test dataset
# test_num = 0
# accuracy_list = list()
# for data in test_loader:
#     x_test, y_test = data
#     output = net(x_test)
#     real= torch.max(y_test,1).indices
#     predict = torch.max(output, 1).indices
#     accuracy = int(sum(real == predict))/len(real)
#     print(accuracy)
#     accuracy_list.append(accuracy)
#     test_num += 1
#     if test_num >= 100:
#         break
# print(f'{test_num}次平均准确率：',torch.mean(torch.tensor(accuracy_list)))


