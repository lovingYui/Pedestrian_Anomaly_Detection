#dataset.py包含了两个数据集类(训练、测试)，以及图像增强处理的方法
import torch
from torch.utils.data import Dataset

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

import os
import glob

class SequenceDataset(Dataset):
    def __init__(self, channels, size, videos_dir, time_steps):

        self.videos = []
        self.seqs_idx = []

        for f in sorted(os.listdir(videos_dir)):
            #print(f)
            frames = glob.glob(os.path.join(videos_dir, f, '*.tif'), recursive=True)
            frames.sort()
            self.videos.append(frames)
            # print(len(frames)) 这里的frames代表的是每个训练集，如Train001的tif文件数目

            selected_idx = np.random.choice(len(frames) - time_steps, size=5)
            #这里-5的原因是防止溢出，size参数代表返回的列表的len

            self.seqs_idx.append(selected_idx)
            #同样，这里的selected_idx是五个随机数一组被打包

        #最后的videos和seqs_idx的len为16，单个元素分别为一个Train/五元随机数组
        self.time_steps = time_steps
        self.size = size

        self.channels = channels
    
    def __len__(self):
        return len(self.videos)
       
    def __getitem__(self, index):

        video = self.videos[index]
        #self.videos为len为16的列表，代表Train001,002...016,这里取index代表一个Train001里头所有的图像
        selected_idx = self.seqs_idx[index]
        #seqs.idx为len为16的列表，每一个元素为五个随机数的列表，即索引下标得到一个包含五个随机数的selected_idx
        clips = []

        for idx in selected_idx:
            frames = video[idx:idx + self.time_steps]
            #从抽取的随机数下标开始，连取五帧图像，然后做训练

            # print(frames)
            #这里的frames代表连续的五个tif文件的路径

            if self.channels == 1:
                frames = [cv2.imread(frame, cv2.IMREAD_GRAYSCALE).astype(np.float32) for frame in frames]
            else:
                frames = [cv2.imread(frame, cv2.IMREAD_COLOR).astype(np.float32)for frame in frames]
            #imread，channels为3，加载彩色图片，cv2.IMREAD_COLOR，也可以直接写1

            # print(torch.tensor(frames).shape)
            #这里的frames形状为tensor(5,240,360,3)

            frames = [simple_transform(frame, self.size, self.channels) for frame in frames]
            #对每一帧图像做处理，这里的frames(transform处理前)长度为5，每个元素为（240,360,3）的array
            
            # print(frames[0].shape)
            #做处理后,这里的frames是一个张量，长度为5，每个元素为torch.size(3,256,256)

            frames = torch.stack(frames)
            # print(torch.tensor(frames).shape)
            #stack处理后为torch.Size([5, 3, 256, 256]),四维，处理之前为一个tensor，里面元素有5个

            frames = torch.reshape(frames,(-1, self.size, self.size))
            # print(torch.tensor(frames).shape) 
            #此时reshape后为torch.Size([15, 256, 256])，转化为三维张量

            clips.append(frames)
            #注意，这里的clips也是一个列表，len为5，列表的每一个元素代表的是以每一个随机下标为起始帧
            #而后连续五帧图像转化成的一个三维张量，shape为torch.Size([15, 256, 256])
            #这里的5就代表的是seqs.idx的长度，从5个不同的起始点开始连续抽取5张图像

        return clips

class TestDataset(Dataset):
    def __init__(self, channels, size, videos_dir, time_steps):
        
        self.videos = glob.glob(os.path.join(videos_dir, '*.tif'), recursive=True)
        #print(self.videos)
        self.videos.sort()

        self.time_steps = time_steps
        self.size = size

        self.channels = channels
    
    def __len__(self):
        #print(len(self.videos))
        #print(len(self.videos) - self.time_steps)
        return len(self.videos) - self.time_steps

    def __getitem__(self, index):

        if self.channels == 1:
            frames = [cv2.imread(frame, cv2.IMREAD_GRAYSCALE).astype(np.float32) for frame in self.videos[index:index + self.time_steps]]
        else:
            frames = [cv2.imread(frame, cv2.IMREAD_COLOR).astype(np.float32) for frame in self.videos[index:index + self.time_steps]]
        #每一次用五张连续帧，共175次

        # print(torch.tensor(frames).shape)  [5,240,360,3]

        o_seqs = [base_transform(img, self.size) for img in frames]
        seqs = [simple_transform(img, self.size, self.channels) for img in frames] #图像增强处理
       
        # print(o_seqs[0].shape)  #o_seqs和seqs为一个长度为5的list,元素均为(3,256,256)
        # print(seqs[0].shape)
        # print(type(o_seqs),len(o_seqs)) #seqs为torch.size,而o_seqs为array类型
        # print(type(seqs),len(seqs))

        seqs = torch.stack(seqs)
        # print(seqs.shape) #同样，stack后seqs变为四维张量[5,3,256,256]
        seqs = torch.reshape(seqs, (-1, self.size, self.size))
        # print(seqs.shape)
        
        return seqs, o_seqs   #返回的seqs为[15,256,256],而o_seqs为一个列表

def simple_transform(img, size, channels):
    if channels == 1:
        mean = 0.5
        std = 0.5
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    
    transform = A.Compose([
        A.Resize(height=size,
                width=size,
                always_apply=True,
                p=1.0),
        A.Normalize(mean=mean,
                    std=std,
                    max_pixel_value=255.0,
                    p=1.0),
        ToTensorV2(p=1.0)
    ], p=0.1)
    img = transform(image=img)['image']

    return img

def base_transform(img, size):
    transform = A.Compose([
        A.Resize(height=size,
                width=size,
                always_apply=True,
                p=1.0)
        ],p=0.1)

    img = transform(image = img)['image']
    return img


