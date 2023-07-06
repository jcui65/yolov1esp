import argparse
import os
import sys#one solution from stackoverflow on how to solve the opencv problem when python2 and 3 coexist
print(sys.path)#not only work for anaconda as stated in the answer
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')#but also work for pycharm
import cv2 as cv
import numpy as np
from models.model_tiny_yolov1 import model_tiny_yolov1ae#models(folder).model_tiny_yolov1(.py file)
from keras.engine import Input
from keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error

parser = argparse.ArgumentParser(description='Use Tiny-Yolov1 To Detect Picture.')
parser.add_argument('weights_path', help='Path to model weights.')
parser.add_argument('image_path', help='Path to detect image.')#got the meaning

classes_name = ['moving']#that is the sequence


class Tiny_Yolov1(object):

    def __init__(self, weights_path, input_path):#through parser, get the weights_path and input_path
        self.weights_path = weights_path#go to self
        self.input_path = input_path#go to self
        self.classes_name = ['moving']#why define it again?
        self.S=10#5
        self.height=3
    def predict(self,index):
        #image = cv.imread(self.input_path)#l26,#its shape is still the original shape
        input_shape = (1,self.height,640+self.height-1,1)#(1, 448, 448, 3)
        #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)#meet the demand of opencv
        #image = cv.resize(image, input_shape[1:3])#resize to 1,2,3, that is 448*448(*3), from 300*500*3
        #image = np.reshape(image, input_shape)#reshape from 448*448*3 into 1*448*448*3
        #image = image / 255.#normalization
        imagepath=np.load(self.input_path)
        #data=np.load('/home/jianning/npylaserdata/data01281528padreshape.npy')
        image=imagepath[index]
        image=np.reshape(image, input_shape)
        inputs = Input(input_shape[1:4])#Keras' Input!#?*448*448*3#for me it is 6*644*1
        outputs = model_tiny_yolov1ae(inputs)#with all the layers, all the parameters
        model = Model(inputs=inputs, outputs=outputs)#This model will include all layers required in the computation of b given a.
        model.load_weights(self.weights_path, by_name=True)
        y = model.predict(image)#, batch_size=1)#do the inference with batch_size=1#y shape 1*7*7*30

        return y


def yolo_head(feats):#then what is feat? just a name? yes, just a parameter that passes in
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = np.shape(feats)[0:1]#[0:2]  # assuming channels last#, just the 2 things/dims
    # In YOLO the height index is the inner most iteration.
    #conv_height_index = np.arange(0, stop=conv_dims[0])#from 0 to 6#non keras version
    conv_width_index = np.arange(0, stop=conv_dims[0])#[1])#do you see conv_dims[2]?
    #conv_height_index = np.tile(conv_height_index, [conv_dims[1]])#tile is copying#it is a machine gun!
    # 0123456012345601234560123456012345601234560123456
    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    #conv_width_index = np.tile(np.expand_dims(conv_width_index, 0), [conv_dims[0], 1])#it is a machine gun!
    #conv_width_index = np.reshape(np.transpose(conv_width_index), [conv_dims[0] * conv_dims[1]])#0000000111111122222223333333
    #conv_index = np.transpose(np.stack([conv_height_index, conv_width_index]))
    #conv_index = np.reshape(conv_index, [conv_dims[0], conv_dims[1], 1, 2])#almost all the above parts are for conv_index
    conv_index = np.reshape(conv_width_index, [conv_dims[0], 1, 1])#511#[conv_dims[0], conv_dims[1], 1, 2])
    conv_dims = np.reshape(conv_dims, [1, 1, 1])#[1, 1, 1, 2])#[[[[7 7]]]]

    #box_xy = (feats[..., :2] + conv_index) / conv_dims * 448#bias+index, times 448/7
    #box_wh = feats[..., 2:4] * 448#directly times 448
    box_x = (feats[..., :1] + conv_index) / conv_dims * 640  # bias+index, times 448/7
    box_w = feats[..., 1:2] * 640  # directly times 448
    #corresponding to the dual operation in data_sequence
    #return box_xy, box_wh#got this function!
    return box_x, box_w#got this function!#return 511,511; or 521,521


def xywh2minmax(xy, wh):#keep the dimension
    xy_min = xy - wh / 2#contains 2 things, horizontal and vertical
    xy_max = xy + wh / 2#from the center of the image to the the upperleft and lower right corner

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):#relatively
    intersect_mins = np.maximum(pred_mins, true_mins)#its shape is 2
    intersect_maxes = np.minimum(pred_maxes, true_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh#intersect_wh[..., 0] #* intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins#should be 2 dim things
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh#[..., 0] #* pred_wh[..., 1]
    true_areas = true_wh#[..., 0] #* true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas#set theory
    iou_scores = intersect_areas / union_areas

    return iou_scores


def _main(args):#this is the function that is explicitly executed
    weights_path = os.path.expanduser(args.weights_path)#see l13 for weights_path
    image_path = os.path.expanduser(args.image_path)# no need for home, since it is just this folder
    time = '3_5_16_5'  # '2_3_11_6'
    gtlabel=np.load('/home/jianning/npylaserdata/data{}sumshufflepadreshape5to3nes.npy'.format(time))
    print('gtlabel shape is:{}'.format(gtlabel.shape))
    tyv1 = Tiny_Yolov1(weights_path, image_path)#tyv1 is a class, with wp and ip as initialization
    S=5
    iouthreshold=0.5
    bmthreshold = 32  # 20#1#
    fpthreshold = 32
    distthreshold = 4  # must be less than (not even equal to 5)#4.9999 is as if this weight does not exist
    bathreshold = 64  # 1#0 means no such weight, but it must be bigger than zero!!! otherwise divided by zero!!!#
    method='yoloae'
    filescenario='reproduce'
    tp = 0
    fn = 0
    fp = 0
    for kk in range(100,150):
        prediction = tyv1.predict(kk)#l32#its shape is 1*7*7*30, then the thing is change to labeled pictures
        predictionr=prediction.reshape((3,642))
        #print('the shape of prediction is: {}'.format(prediction.shape))#(1, 3, 642, 1)
        thislabel=gtlabel[kk].reshape((3,642))#(3, 642, 1) before reshaping
        #print(thislabel.shape)
        #print('the shape of prediction is: {}'.format(predictionr))  # (1, 3, 642, 1) before reshaping
        mse = mean_squared_error(thislabel, predictionr)
        print('the mse is: {}'.format(mse))
        mae = mean_absolute_error(thislabel, predictionr)
        print('the mae is: {}'.format(mae))
        '''
        tp1 = 0
        fn1 = 0
        fp1 = 0
        for i in range(int(len(msga) / 2)):
            lb = int(len(msgb) / 2)
            for j in range(lb):
                iouvalue = iou(msgb[2 * j], msgb[2 * j + 1], msga[2 * i], msga[2 * i + 1])
                if iouvalue > iouthreshold:  # 0.6:#
                    tp1 = tp1 + 1
                    deleteindex = [2 * j, 2 * j + 1]
                    msgb = np.delete(msgb, deleteindex)
                    break
            if lb == len(msgb) / 2:  #:
                #weight1 = (msga[2 * i + 1] - msga[2 * i] + 0.0) / bmthreshold
                weight1 =1# min(weight1, 1.0)
                #rangecenter = (msgaranges[msga[2 * i + 1]] + msgaranges[msga[2 * i]]) / 2.0
                weight2 = 1#max(min(1, -1.0 / (5 - distthreshold) * (rangecenter - 5)), 0)
                #bac = (msga[2 * i + 1] + msga[2 * i]) / 2  # bac for bearing angle center
                weight3 = 1#max(min((320 - np.absolute(bac - 320)) / bathreshold, 1), 0)
                weight = weight1 * weight2 * weight3
                #print('The weight is:', weight)
                fn1 = fn1 + 1 * min(weight, 1)
            # deleteindexgt = [2 * i, 2 * i + 1]
            # msga = np.delete(msga, deleteindexgt)
        if len(msgb) / 2 > 0:
            for i in range(int(len(msgb) / 2)):
                #weight4 = (msgb[2 * i + 1] - msgb[2 * i] + 0.0) / fpthreshold
                weight4 = 1#min(weight4, 1)
                #print('The weight fp is:', weight4)
                fp1 = fp1 + weight4  # should be ok, right?
            # fp1=fp1+len(msgb)/2#here there should be some changes
        tp = tp + tp1
        fn = fn + fn1
        fp = fp + fp1

    print('false positive is:', fp)
    print('false negative is:', fn)
    print('true positive is:', tp)
    #if (tp + fp) == 0 or (tp + fn) == 0 or (fp + fn + tp) == 0:
        #break
    acc = (tp + 0.0) / (fp + fn + tp)  # acc for accuracy
    pre = tp / (tp + fp + 0.0)  # pre for precision
    rec = tp / (tp + fn + 0.0)  # rec for recall
    #if pre == 0 and rec == 0:
        #break
    f1score = 2.0 * pre * rec / (pre + rec)
    print('accuracy is:', acc)
    print('precision is:', pre)
    print('recall is:', rec)
    print('f1 score is:', f1score)
    filepath = '/home/jianning/cmd_record/CuiJ_MS/{}results/data/'.format(method)  # the_right_results/
    filename = filepath + filescenario
    f = open(filename, "w")
    # f.write("Woops! I have deleted the content!")
    # f.write('\nthe total number of points: {}'.format(msglcgt.shape[0])+'\n\n')
    # f.write('the confusion matrix:\n {}\n\n'.format(confusion_matrix(ytruer, ypr)))
    # f.write('true negative: {}'.format(tn)+'\t')
    f.write('iou threshold: {}'.format(iouthreshold) + '\n\n')
    f.write('false positive: {}'.format(fp) + '\n')
    f.write('false negative: {}'.format(fn) + '\t')
    f.write('true positive: {}'.format(tp) + '\n\n')
    f.write('accuracy is:  {}'.format(acc) + '\n')
    f.write('precision is: {}'.format(pre) + '\n')
    f.write('the recall is:{}'.format(rec) + '\n')
    f.write('f1 score is:  {}'.format(f1score))
    f.close()
    resultnp = np.array([fp, fn, tp])
    method='gt14205'
    filepathnpy = '/home/jianning/cmd_record/CuiJ_MS/{}results/npyfile/'.format(method)  # the_right_npy/
    npyname = filepathnpy + filescenario + '.npy'
    np.save(npyname, resultnp)'''




if __name__ == '__main__':
    #_main(parser.parse_args())
    time='3_5_16_5'#'2_3_11_6'
    #_main(parser.parse_args(["checkpoints/model41702aen20p20small5to3.h5", '/home/jianning/npylaserdata/data{}sumshufflepadreshape5to3nes.npy'.format(time)]))#data01281528padreshape.npy']))#'VOCdevkit/VOC2007/JPEGImages/000015.jpg']))
    _main(parser.parse_args(["checkpoints/model41702aen150p20small5to3tanh.h5",
                             '/home/jianning/npylaserdata/data{}sumshufflepadreshape5to3nes.npy'.format(
                                 time)]))
