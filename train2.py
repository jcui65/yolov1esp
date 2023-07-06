import argparse
import keras
from keras.engine import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import os
from models.model_tiny_yolov1 import model_tiny_yolov1,model_tiny_yolov12,model_tiny_yolov122,model_tiny_yolov1tanh,model_tiny_yolov123
from data_sequence import SequenceData
from yolo.yolo import yolo_loss#should be changed
from callback import callback
import numpy as np
parser = argparse.ArgumentParser(description='Train NetWork.')
parser.add_argument('epochs', help='Num of epochs.')
parser.add_argument('batch_size', help='Num of batch size.')
parser.add_argument('datasets_path', help='Path to datasets.')
#follow this example and use it
#edited by Jianning Cui
from models.model_tiny_yolov1 import Yolo_Reshape
from keras.engine.topology import Layer
import datetime


#end of edited by Jianning Cui




def _main(args):#that function is called _main
    S=10#5#
    trainamount=37500#12000#
    epochs = int(os.path.expanduser(args.epochs))#should be what I entered, yet notice the priority, see line 12
    batch_size = int(os.path.expanduser(args.batch_size))#line 13
    ih=3#5#ablation study #5#original
    input_shape = (ih,640+ih-1,1)#(448, 448, 3)#from numpy array to tensor
    inputs = Input(input_shape)
    yolo_outputs = model_tiny_yolov123(inputs)#model_tiny_yolov1tanh(inputs)#line 6
    
    model = Model(inputs=inputs, outputs=yolo_outputs)
    model.summary()
    print(model.trainable_weights)
    adam1 = keras.optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #sgd = keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=False)
    sgd = keras.optimizers.SGD(lr=0.00008, decay=1e-6, momentum=0.9, nesterov=False)
    #sgd = keras.optimizers.SGD(lr=0.000005, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss=yolo_loss, optimizer=sgd)#adam1)
    '''
    save_dir = 'checkpoints'
    weights_path = os.path.join(save_dir, 'weightses.hdf5')#it means to create a folder called checkpoints
    '''
    save_dir = 'checkpoints'
    model_path = os.path.join(save_dir, 'model41702n10rp{epoch:02d}small5to3aetanhnewpretrainsgd.h5')  # it means to create a folder called checkpoints
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss',verbose=1,
                                 save_weights_only=False, save_best_only=True,period=2)
    #if not os.path.isdir(save_dir):#one step at a time, OK?
        #os.makedirs(save_dir)#if there is no, then create one just like that
    filenameweights = 'checkpoints/model41702n32p10small5to3aetanhnewpretrainsgd.h5'
    #model41702n32p08small5to3aetanhpretrainsgd.h5'#model41702n14small5to3aetanhnopretrainsgd.h5'#'model-tiny-yolov1ese11111s10d28105.h5'
    if os.path.exists(filenameweights):#('checkpoints/weights.hdf5'):
        #model=keras.models.load_model(filenameweights, custom_objects={'Yolo_Reshape' : Yolo_Reshape})  # ('checkpoints/weights.hdf5', by_name=True)
        model.load_weights(filenameweights, by_name=True)#('checkpoints/weights.hdf5', by_name=True)
        print('has trained for 10 epochs!')#can be load weights, or load the whole model!
    #if os.path.exists('checkpoints/weights.hdf5'):#('checkpoints/weights.hdf5'):
        #model.load_weights('checkpoints/weights.hdf5', by_name=True)#('checkpoints/weights.hdf5', by_name=True)
    else:#load, train, save, load, train, save......maybe that is why it takes time more than the training process
        #model.load_weights('weights/tiny-yolov1.hdf5', by_name=True)#pretrained model to start the training?
        print('no train history')#is it using the existing pretrained weights? Yes!


    #if os.path.exists(filenameweights):#('checkpoints/weights.hdf5'):
    #model=keras.models.load_model(filenameweights, custom_objects={'Yolo_Reshape' : Yolo_Reshape, 'yolo_loss':yolo_loss})  # ('checkpoints/weights.hdf5', by_name=True)
    #model.load_weights(filenameweights, by_name=True)#('checkpoints/weights.hdf5', by_name=True)
    #model.summary()
    #model.compile(loss=yolo_loss, optimizer='adam')
    #print('has trained for 7 epochs!')#can be load weights, or load the whole model!
    #if os.path.exists('checkpoints/weights.hdf5'):#('checkpoints/weights.hdf5'):
        #model.load_weights('checkpoints/weights.hdf5', by_name=True)#('checkpoints/weights.hdf5', by_name=True)
    #else:#load, train, save, load, train, save......maybe that is why it takes time more than the training process
        #model.load_weights('weights/tiny-yolov1.hdf5', by_name=True)#pretrained model to start the training?
        #print('no train history')#is it using the existing pretrained weights? Yes!
        #once it is launched, it will be anoter story
    #epoch_file_path = 'checkpoints/epoch.txt'
    #try:
        #with open(epoch_file_path, 'r') as f:
            #now_epoch = int(f.read())
        #epochs -= now_epoch
    #except IOError:
        #print('no train history')
    #myCallback = callback.MyCallback()

    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    # does it mean it will stop when the validation loss doesn't go down? Almost yes, yet with some patience!!!
    # log_dir = 'logs'
    # tbCallBack = TensorBoard(log_dir=log_dir,  # log directory
    #                          histogram_freq=0,  # calculate histogram according to epoch, 0 means don't calculate
    #                          batch_size=batch_size,  # the amount of data to calculate the historgram
    #                          write_graph=True,  # whether to store the network structure
    #                          write_grads=True,  # whether to visualize the gradient histogram
    #                          write_images=True,  # whether to visualize the parameters
    #                          embeddings_freq=0,
    #                          embeddings_layer_names=None,
    #                          embeddings_metadata=None)
    # if not os.path.isdir(log_dir):
    #     os.makedirs(log_dir)
    # give it to the variable datasets_path
    datasets_path = os.path.expanduser(args.datasets_path)#from strings to parameters
    daterecorded='3_5_16_5'#'2_3_11_6'#'3_4_15_18'
    #traindata = np.load('/home/jianning/npylaserdata/data{}sumshufflepadreshapen.npy'.format(daterecorded))  #
    traindata = np.load('/home/jianning/npylaserdata/data{}sumshufflepadreshape5to3nes.npy'.format(daterecorded))  #
    #traindata=np.load('/home/jianning/npylaserdata/data{}_{}_{}sumshufflepadreshapereally.npy'.format(time))#np.load('/home/jianning/npylaserdata/data01281528padreshape.npy')
    trainlabel = np.load('/home/jianning/npylaserdata/label{}sumshuffleodreshape.npy'.format(daterecorded))
    #trainlabel=np.load('/home/jianning/npylaserdata/label2_3_11_6sumshufflereshape.npy')#np.load('/home/jianning/npylaserdata/label01281528odreshape.npy')
    #traindata = np.load('/home/jianning/npylaserdata/data{}sumshufflepadreshape5to3nes.npy'.format(daterecorded))#nes for normalized error scan

    #trainlabel = np.load('/home/jianning/npylaserdata/label{}sumshuffleodreshape.npy'.format(daterecorded))
    valdata=traindata[trainamount:]
    vallabel=trainlabel[trainamount:]
    traindata=traindata[:trainamount]
    trainlabel=trainlabel[:trainamount]
    #get the dataset path
    # data generator,#the type of train_generator is <class 'data_sequence.SequenceData'>
    #train_generator = SequenceData(#'train' is the model, datasets_path is the dir! input_shape is the target_size
        #'train', datasets_path, input_shape, batch_size)#input_shape 448*448*3,#length of train_generator 79 when bs=32
    #validation_generator = SequenceData(
        #'val', datasets_path, input_shape, batch_size)

    model.fit(#Trains the model on data generated batch-by-batch by a Python generator (or an instance of Sequence).
        traindata,#train_generator,#thus, it is actually a class, not just a batch of images? Sort of?
        trainlabel,#steps_per_epoch=len(train_generator),#when batchsize=32, it is 79.64-40, __len__!!!
        batch_size=1,
        epochs=10,#20,#epochs,#30,#you can comment this line and then pass in the parameter by yourself#verbose=1 = progress bar
        validation_data=(valdata,vallabel),##validation_generator,#you just found how to pass the epoch from parser!!!
        #validation_steps=len(validation_generator),#seems unspecified
        # use_multiprocessing=True,
        #workers=4,
        callbacks=[checkpoint, early_stopping]#see a dozens of lines earlier
    )
    model.save('model-tiny-yolov1ese50s10d41702nes5to3tanhnewpretrainsgd.h5')#model.save_weights('my-tiny-yolov1ese3data14205.h5')


if __name__ == '__main__':
    #_main(parser.parse_args())
    _main(parser.parse_args(['4', '32', '/home/jianning/yolov1/yolov1-keras-voc/VOCdevkit']))
    #the passing of the parameter is only valid when there is no default value of epochs