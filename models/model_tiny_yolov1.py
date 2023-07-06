from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Dense, Reshape, LeakyReLU#, Tanh
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.engine.topology import Layer
import keras.backend as K
import keras
from keras.models import Model


class Yolo_Reshape(Layer):
    def __init__(self, **kwargs):#target_shape,
        super(Yolo_Reshape, self).__init__(**kwargs)
        self.S=10#5#yolodata1#10#old benchmarking
        self.target_shape = (1,self.S,7)#(1,5,7)#tuple(target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape

    def call(self, inputs, **kwargs):
        S = self.S#5#[self.target_shape[0], self.target_shape[1]]
        C = 1#20
        B = 2
        idx1 = S*C#S[0] * S[1] * C
        idx2 = idx1 + S * B#idx1 + S[0] * S[1] * B
        # class prediction
        class_probs = K.reshape(
            inputs[:, :idx1], (K.shape(inputs)[0],) + tuple([S , C]))#tuple([S[0], S[1], C]))
        class_probs = K.softmax(class_probs)
        # confidence
        confs = K.reshape(
            inputs[:, idx1:idx2], (K.shape(inputs)[0],) + tuple([S, B]))#tuple([S[0], S[1], B]))
        confs = K.sigmoid(confs)
        # boxes
        boxes = K.reshape(
            inputs[:, idx2:], (K.shape(inputs)[0],) + tuple([S, B * 2]))#tuple([S[0], S[1], B * 4]))
        boxes = K.sigmoid(boxes)
        # return np.array([class_probs, confs, boxes])
        outputs = K.concatenate([class_probs, confs, boxes])
        return outputs


def model_tiny_yolov1(inputs):
    ih=3#5#initial height
    S=5#10#
    x = Conv2D(16, (ih, ih), padding='valid', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=True)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)

    x = Conv2D(32, (1, 3), padding='same', name='convolutional_1', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True)(x)
    x = BatchNormalization(name='bnconvolutional_1', trainable=True)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)

    x = Conv2D(64, (1, 3), padding='same', name='convolutional_2', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True)(x)
    x = BatchNormalization(name='bnconvolutional_2', trainable=True)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)

    x = Conv2D(128, (1, 3), padding='same', name='convolutional_3', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True)(x)
    x = BatchNormalization(name='bnconvolutional_3', trainable=True)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)
    '''
    x = Conv2D(256, (1, 3), padding='same', name='convolutional_4', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True)(x)
    x = BatchNormalization(name='bnconvolutional_4', trainable=True)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)

    x = Conv2D(512, (1, 3), padding='same', name='convolutional_5', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True)(x)
    x = BatchNormalization(name='bnconvolutional_5', trainable=True)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)

    x = Conv2D(1024, (1, 3), padding='same', name='convolutional_6', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True)(x)
    x = BatchNormalization(name='bnconvolutional_6', trainable=True)(x)
    x = LeakyReLU(alpha=0.1)(x)
    if S==5:
        x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)
    elif S==10:
        pass
    
    x = Conv2D(256, (1, 3), padding='same', name='convolutional_7', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True)(x)
    x = BatchNormalization(name='bnconvolutional_7', trainable=True)(x)
    x = LeakyReLU(alpha=0.1)(x)
    '''
    x = Flatten()(x)
    x = Dense(7*S, activation='linear', name='connected_0')(x)#35
    # outputs = Reshape((7, 7, 30))(x)#5*7;7=1+3*2;3=2+1
    outputs = Yolo_Reshape()(x)#(1, 5, 7)#Yolo_Reshape((7, 7, 30))(x)

    return outputs


def model_tiny_yolov1ae(inputs):
    ih = 3  # 5#initial height
    S = 5  # 10#
    x = Conv2D(16, (ih, ih), padding='valid', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True,activation='tanh')(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=True)(x)
    #x = Tanh(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)

    x = Conv2D(32, (1, 3), padding='same', name='convolutional_1', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True,activation='tanh')(x)
    x = BatchNormalization(name='bnconvolutional_1', trainable=True)(x)
    #x = Tanh(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)

    x = Conv2D(64, (1, 3), padding='same', name='convolutional_2', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True,activation='tanh')(x)
    x = BatchNormalization(name='bnconvolutional_2', trainable=True)(x)
    #x = Tanh(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)

    x = Conv2D(128, (1, 3), padding='same', name='convolutional_3', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True,activation='tanh')(x)
    x = BatchNormalization(name='bnconvolutional_3', trainable=True)(x)
    #x = Tanh(x)
    encoded = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)
    #print(encoded.shape)
    x = Conv2D(128, (1, 3), padding='same', name='convolutional_3r', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True,activation='tanh')(encoded)
    #x = BatchNormalization(name='bnconvolutional_3', trainable=True)(x)
    #x = Tanh(x)
    #print(x.shape)
    x=UpSampling2D((1,2))(x)
    #print(x.shape)
    x = Conv2D(64, (1, 3), padding='same', name='convolutional_2r', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True,activation='tanh')(x)
    # x = BatchNormalization(name='bnconvolutional_3', trainable=True)(x)
    #print(x.shape)
    #x = Tanh(x)
    x = UpSampling2D((1, 2))(x)
    #print(x.shape)
    x = Conv2D(32, (1, 3), padding='same', name='convolutional_1r', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True,activation='tanh')(x)
    # x = BatchNormalization(name='bnconvolutional_3', trainable=True)(x)
    #print(x.shape)
    #x = Tanh(x)
    x = UpSampling2D((1, 2))(x)
    #print(x.shape)
    x = Conv2D(16, (1, 3), padding='same', name='convolutional_0r', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True,activation='tanh')(x)
    # x = BatchNormalization(name='bnconvolutional_3', trainable=True)(x)
    #print(x.shape)
    #x = Tanh(x)
    x = UpSampling2D((1, 2))(x)
    #print(x.shape)
    x = Conv2DTranspose(1,(ih, ih), padding='valid', name='convT_back', use_bias=False, kernel_regularizer=l2(5e-4), trainable=True,activation='tanh')(x)
    #print(x.shape)
    #x=Conv2D(1, (ih, ih), padding='same', name='convolutional_0', use_bias=False,
               #kernel_regularizer=l2(5e-4), trainable=True)(inputs)
    decoded = x#Tanh(x)
    #print(decoded.shape)
    #x = Flatten()(x)
    #x = Dense(7 * S, activation='linear', name='connected_0')(x)  # 35
    # outputs = Reshape((7, 7, 30))(x)#5*7;7=1+3*2;3=2+1
    outputs = decoded#Yolo_Reshape()(x)  # (1, 5, 7)#Yolo_Reshape((7, 7, 30))(x)

    return outputs


def model_tiny_yolov12(inputs):
    ih = 3  # 5#initial height
    S = 10  # 10#
    autoencoder=keras.models.load_model("checkpoints/model41702aen150p20small5to3tanh.h5")
    encoder=Model(autoencoder.input,autoencoder.layers[-15].output)
    encoder.summary()
    x=encoder(inputs)#(inputs=inputs)
    x = Flatten()(x)
    x = Dense(7 * S, activation='linear', name='connected_0')(x)  # 35
    # outputs = Reshape((7, 7, 30))(x)#5*7;7=1+3*2;3=2+1
    outputs = Yolo_Reshape()(x)  # (1, 5, 7)#Yolo_Reshape((7, 7, 30))(x)

    return outputs

def model_tiny_yolov122(inputs):
    ih = 3  # 5#initial height
    S = 10  # 10#
    autoencoder=keras.models.load_model("checkpoints/model41702aen150p20small5to3tanh.h5")
    encoder=Model(autoencoder.input,autoencoder.layers[-15].output)
    encoder.trainable=False
    encoder.summary()
    x=encoder(inputs)#(inputs=inputs)
    x = Flatten()(x)
    x = Dense(7 * S, activation='linear', name='connected_0')(x)  # 35
    # outputs = Reshape((7, 7, 30))(x)#5*7;7=1+3*2;3=2+1
    outputs = Yolo_Reshape()(x)  # (1, 5, 7)#Yolo_Reshape((7, 7, 30))(x)

    return outputs

def model_tiny_yolov123(inputs):
    ih = 3  # 5#initial height
    S = 10  # 10#
    autoencoder=keras.models.load_model("checkpoints/model41702aen40rp20small5to3tanhadamfast.h5")
    encoder=Model(autoencoder.input,autoencoder.layers[-15].output)
    encoder.trainable=False
    encoder.summary()
    x=encoder(inputs)#(inputs=inputs)
    x = Flatten()(x)
    x = Dense(7 * S, activation='linear', name='connected_0')(x)  # 35
    # outputs = Reshape((7, 7, 30))(x)#5*7;7=1+3*2;3=2+1
    outputs = Yolo_Reshape()(x)  # (1, 5, 7)#Yolo_Reshape((7, 7, 30))(x)

    return outputs


def model_tiny_yolov1tanh(inputs):
    ih = 3  # 5#initial height
    S = 10#5  #
    x = Conv2D(16, (ih, ih), padding='valid', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True,activation='tanh')(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=True)(x)
    #x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)

    x = Conv2D(32, (1, 3), padding='same', name='convolutional_1', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True,activation='tanh')(x)
    x = BatchNormalization(name='bnconvolutional_1', trainable=True)(x)
    #x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)

    x = Conv2D(64, (1, 3), padding='same', name='convolutional_2', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True,activation='tanh')(x)
    x = BatchNormalization(name='bnconvolutional_2', trainable=True)(x)
    #x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)

    x = Conv2D(128, (1, 3), padding='same', name='convolutional_3', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True,activation='tanh')(x)
    x = BatchNormalization(name='bnconvolutional_3', trainable=True)(x)
    #x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)
    '''
    x = Conv2D(256, (1, 3), padding='same', name='convolutional_4', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True)(x)
    x = BatchNormalization(name='bnconvolutional_4', trainable=True)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)

    x = Conv2D(512, (1, 3), padding='same', name='convolutional_5', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True)(x)
    x = BatchNormalization(name='bnconvolutional_5', trainable=True)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)

    x = Conv2D(1024, (1, 3), padding='same', name='convolutional_6', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True)(x)
    x = BatchNormalization(name='bnconvolutional_6', trainable=True)(x)
    x = LeakyReLU(alpha=0.1)(x)
    if S==5:
        x = MaxPooling2D((1, 2), strides=(1, 2), padding='same')(x)
    elif S==10:
        pass

    x = Conv2D(256, (1, 3), padding='same', name='convolutional_7', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=True)(x)
    x = BatchNormalization(name='bnconvolutional_7', trainable=True)(x)
    x = LeakyReLU(alpha=0.1)(x)
    '''
    x = Flatten()(x)
    x = Dense(7 * S, activation='linear', name='connected_0')(x)  # 35
    # outputs = Reshape((7, 7, 30))(x)#5*7;7=1+3*2;3=2+1
    outputs = Yolo_Reshape()(x)  # (1, 5, 7)#Yolo_Reshape((7, 7, 30))(x)

    return outputs