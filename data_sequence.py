from keras.utils import Sequence
import math
import sys#one solution from stackoverflow on how to solve the opencv problem when python2 and 3 coexist
print(sys.path)#not only work for anaconda as stated in the answer
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')#but also work for pycharm
import cv2 as cv#check if it is using python2 or 3
import numpy as np
import os


class SequenceData(Sequence):

    def __init__(self, model, dir, target_size, batch_size, shuffle=True):
        self.model = model#here the dir is the path of the dataset!
        self.datasets = []#I think I can write the following code more concise, using .format() method!
        #if self.model is 'train':#VOCdevkit/2007_train.txt
            #with open(os.path.join(dir, '2007_train.txt'), 'r') as f:#concatenating file directory! read mode
                #self.datasets = self.datasets + f.readlines()#do you want to print it? 2501 lines?
        #elif self.model is 'val':
        with open(os.path.join(dir, '2007_{}.txt'.format(self.model)), 'r') as f:#open a file and read each line
            self.datasets = self.datasets + f.readlines()#the photo info/errorscan photo info+bounding boxes
        self.image_size = target_size[0:2]#448*448, for me, it may be 640/320*3/4/5/6/7
        self.batch_size = batch_size#32?1?whatever?
        self.indexes = np.arange(len(self.datasets))#from 1 to 2501. for me, hehe, 10k+ is prefereed
        self.shuffle = shuffle#if default is true, then it is true

    def __len__(self):
        # 计算每一个epoch的迭代次数
        num_imgs = len(self.datasets)#need to print to see what it is, 79? 79*32? Now I think it is 2501.
        return math.ceil(num_imgs / float(self.batch_size))#2501/32, then ceil, equals 79

    def __getitem__(self, idx):
        # 生成batch_size个索引
        batch_indexs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]#slicing!
        # 根据索引获取datas集合中的数据
        batch = [self.datasets[k] for k in batch_indexs]#32ge, batch size ge
        # 生成数据
        X, y = self.data_generation(batch)#see a few dozen lines later!
        return X, y#32 image names/info, 32 label info

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def read(self, dataset):#the normalization process is done in read member function
        dataset = dataset.strip().split()#get rid of the spaces on both ends, then break into pieces, a list now!
        image_path = dataset[0]#the 0th piece is no doubt the image_path
        label = dataset[1:]#how to deal with the variable length problem?
        #an example: [/home/jianning/yolov1/yolov1-keras-voc/VOCdevkit/VOC2007/JPEGImages/000001.jpg&48,240,195,371,8&8,12,352,498,9]
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # opencv读取通道顺序为BGR，所以要转换
        image_h, image_w = image.shape[0:2]#0 and 1, not including 2
        image = cv.resize(image, self.image_size)#how to resize? Not crop, but dilate/magnify/shrink
        image = image / 255.#normalize

        label_matrix = np.zeros([7, 7, 25])#20 categroies, 1 confidence, 4 box data
        for l in label:#each l is like: 48,240,195,371,8; 8,12,352,498,9
            l = l.split(',')#now understand!
            l = np.array(l, dtype=np.int)#they come in 4
            xmin = l[0]
            ymin = l[1]
            xmax = l[2]
            ymax = l[3]
            cls = l[4]
            x = (xmin + xmax) / 2 / image_w#get the percentage of the square
            y = (ymin + ymax) / 2 / image_h#I have found the statement in the original paper!!!
            w = (xmax - xmin) / image_w#normalized width
            h = (ymax - ymin) / image_h#normalized height
            loc = [7 * x, 7 * y]#3.8,5.7
            loc_i = int(loc[1])
            loc_j = int(loc[0])
            y = loc[1] - loc_i#bias, shift, pianyiliang, go to see the original paper!
            x = loc[0] - loc_j#confirmed from the original paper!

            if label_matrix[loc_i, loc_j, 24] == 0:#must go through
                label_matrix[loc_i, loc_j, cls] = 1#sparse representation
                label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
                label_matrix[loc_i, loc_j, 24] = 1  # response

        return image, label_matrix#each label is a label matrix of size 7*7*25, for me it is 5*4(4=1+3,3=2+1)

    def data_generation(self, batch_datasets):
        images = []
        labels = []

        for dataset in batch_datasets:#dataset is a single line
            image, label = self.read(dataset)#the above function, line 46
            images.append(image)
            labels.append(label)

        X = np.array(images)
        y = np.array(labels)

        return X, y
