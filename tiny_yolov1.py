import argparse
import os
import sys#one solution from stackoverflow on how to solve the opencv problem when python2 and 3 coexist
print(sys.path)#not only work for anaconda as stated in the answer
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')#but also work for pycharm
import cv2 as cv
import numpy as np
from models.model_tiny_yolov1 import model_tiny_yolov1#models(folder).model_tiny_yolov1(.py file)
from keras.engine import Input
from keras.models import Model

parser = argparse.ArgumentParser(description='Use Tiny-Yolov1 To Detect Picture.')
parser.add_argument('weights_path', help='Path to model weights.')
parser.add_argument('image_path', help='Path to detect image.')#got the meaning

classes_name = ['moving']#that is the sequence


class Tiny_Yolov1(object):

    def __init__(self, weights_path, input_path):#through parser, get the weights_path and input_path
        self.weights_path = weights_path#go to self
        self.input_path = input_path#go to self
        self.classes_name = ['moving']#why define it again?
        self.S=5

    def predict(self):
        #image = cv.imread(self.input_path)#l26,#its shape is still the original shape
        input_shape = (1,self.S,644,1)#(1, 448, 448, 3)
        #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)#meet the demand of opencv
        #image = cv.resize(image, input_shape[1:3])#resize to 1,2,3, that is 448*448(*3), from 300*500*3
        #image = np.reshape(image, input_shape)#reshape from 448*448*3 into 1*448*448*3
        #image = image / 255.#normalization
        imagepath=np.load(self.input_path)
        #data=np.load('/home/jianning/npylaserdata/data01281528padreshape.npy')
        image=imagepath[3]
        image=np.reshape(image, input_shape)
        inputs = Input(input_shape[1:4])#Keras' Input!#?*448*448*3#for me it is 6*644*1
        outputs = model_tiny_yolov1(inputs)#with all the layers, all the parameters
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
    intersect_areas = intersect_wh[..., 0] #* intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins#should be 2 dim things
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] #* pred_wh[..., 1]
    true_areas = true_wh[..., 0] #* true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas#set theory
    iou_scores = intersect_areas / union_areas

    return iou_scores


def _main(args):#this is the function that is explicitly executed
    weights_path = os.path.expanduser(args.weights_path)#see l13 for weights_path
    image_path = os.path.expanduser(args.image_path)# no need for home, since it is just this folder

    tyv1 = Tiny_Yolov1(weights_path, image_path)#tyv1 is a class, with wp and ip as initialization
    prediction = tyv1.predict()#l32#its shape is 1*7*7*30, then the thing is change to labeled pictures

    predict_class = prediction[..., :1]#20]  # 1 * 7 * 7 * 20,#20 class probability#151
    predict_trust = prediction[..., 1:3]#20:22]  # 1 * 7 * 7 * 2#there are 2 boxes#152
    predict_box = prediction[..., 3:]#22:]  # 1 * 7 * 7 * 8#the x,y,w,h for the 2 boxes#154

    predict_class = np.reshape(predict_class, [self.S,1,1])#[7, 7, 1, 20])# for each grid cell
    predict_trust = np.reshape(predict_trust, [self.S,2,1])#[7, 7, 2, 1])#the confidence value of the 2 boxes
    predict_box = np.reshape(predict_box, [self.S,2,2])#[7, 7, 2, 4])#the reshape is for the following product

    predict_scores = predict_class * predict_trust  # 7 * 7 * 2 * 20#(1) in paper, for each box, not cell#for me,521(the last 1 is moving)

    box_classes = np.argmax(predict_scores, axis=-1)  # 7 * 7 * 2#the index, the class for each box
    box_class_scores = np.max(predict_scores, axis=-1)  # 7 * 7 * 2#the score
    best_box_class_scores = np.max(box_class_scores, axis=-1, keepdims=True)  # 7 * 7 * 1# find the best box#771 is why keepdims

    box_mask = box_class_scores >= best_box_class_scores  # ? * 7 * 7 * 2#>= is just geq, get the right box

    filter_mask = box_class_scores >= 0.6  # 7 * 7 * 2#0.4#0.706#not only the best among the 2, but must be good enough
    filter_mask *= box_mask  # 7 * 7 * 2#2 filters, sorting from the best boxes for good enough

    filter_mask = np.expand_dims(filter_mask, axis=-1)  # 7 * 7 * 2 * 1# from 772 to 7 * 7 * 2 * 1#521
    # nothing physical, but just get one more new dimension
    predict_scores *= filter_mask  # 7 * 7 * 2 * 20#see l105#get rid of those bad scores#find the best and goodenough box#521,2 for 2boxes?1 for class moving
    predict_box *= filter_mask  # 7 * 7 * 2 * 4#get rid of the coordinates of those badenough boxes#522, 2 for 2 boxes, 2 for xw

    box_classes = np.expand_dims(box_classes, axis=-1)#7721#52to521
    box_classes *= filter_mask  # 7 * 7 * 2 * 1#, goodenough#521
    # the center/the bias 7,7,2,2, each box, vertical & horizontal#the width and the height 7,7,2,2, each box, width and height
    box_x, box_w = yolo_head(predict_box)  # 7 * 7 * 2 * 2#each, feeding in 7724, 4 for xywh goodenough#5*2*1, 5*2*1
    box_x_min, box_x_max = xywh2minmax(box_x, box_w)  # 7 * 7 * 2 * 2#521,521
    # the upperleft corner of the box, 7,7,2,2#the lowerright corner of the box,7722
    predict_trust *= filter_mask  # 7 * 7 * 2 * 1#, nothing concrete, no need to talk about things#521
    nms_mask = np.zeros_like(filter_mask)  # 7 * 7 * 2 * 1#boxwise, thus 2#521
    predict_trust_max = np.max(predict_trust)  # 找到置信度最高的框#find 1 from 2*goodenough (all)##only one value!!!
    max_i = max_k = 0#max_j=#find the box with the highest confidence/IOU #what are these? now I know!!!
    while predict_trust_max > 0:#it is the NMS PROCEDURE!!!
        for i in range(nms_mask.shape[0]):#5
            for k in range(nms_mask.shape[1]):#2
                if predict_trust[i, k, 0] == predict_trust_max:
                    nms_mask[i, k, 0] = 1
                    filter_mask[i, k, 0] = 0
                    max_i = i
                    #max_j = j
                    max_k = k
        for i in range(nms_mask.shape[0]):#5
            #for j in range(nms_mask.shape[1]):#
            for k in range(nms_mask.shape[2]):#2
                if filter_mask[i, k, 0] == 1:
                    iou_score = iou(box_x_min[max_i, max_k, :],#max_j,
                                    box_x_max[max_i, max_k, :],#max_j,
                                    box_x_min[i, k, :],#j,
                                    box_x_max[i, k, :])#j,
                    if iou_score > 0.4:#since it is 1-d, probably 0.4 is better than 0.2#0.2:
                        filter_mask[i,  k, 0] = 0#j,
        predict_trust *= filter_mask  # 7 * 7 * 2 * 1#cut some of the high overlapped boxes#for me 521
        predict_trust_max = np.max(predict_trust)  # 找到置信度最高的框

    box_x_min *= nms_mask
    box_x_max *= nms_mask
    #data = np.load('/home/jianning/npylaserdata/data01281528padreshape.npy')
    #image = data[0]
    #image = #cv.imread(image_path)#prepare to label the inference results
    #origin_shape = image.shape[0:2]#for me it is still right!
    #image = cv.resize(image, (448, 448))
    detect_shape = filter_mask.shape

    for i in range(detect_shape[0]):#5
        #for j in range(detect_shape[1]):
        for k in range(detect_shape[1]):#2
            if nms_mask[i, k, 0]:#j,
                print('the boxmin is:',int(box_x_min[i, k, 0]))
                print('the boxmax is:', int(box_x_max[i, k, 0]))
                #cv.rectangle(image, (int(box_x_min[i, k, 0]),1),# int(box_x_min[i, j, k, 1])),
                            #(int(box_x_max[i, k, 0]),4), #int(box_xy_max[i, j, k, 1])),
                            #(0, 0, 255))
                #cv.putText(image, classes_name[box_classes[i, j, k, 0]],
                            #(int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                            #1, 1, (0, 0, 255))

    #image = cv.resize(image, (origin_shape[1], origin_shape[0]))
    #cv.imshow('image', image)
    #cv.waitKey(0)


if __name__ == '__main__':
    #_main(parser.parse_args())
    _main(parser.parse_args(['my-tiny-yolov1ese2.hdf5', '/home/jianning/npylaserdata/data01281528padreshape.npy']))#'VOCdevkit/VOC2007/JPEGImages/000015.jpg']))
