import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

#Define the model
def get_ARCNN(input_shape=(32,32,1)):
    inp = Input(shape=input_shape)
    conv1 = Conv2D(64,9,activation='relu', padding='same', use_bias=True,name="Feature_extract")(inp)
    conv2 = Conv2D(32,7,activation='relu', padding='same', use_bias=True,name="Feature_Enhance")(conv1)
    conv3 = Conv2D(64,1,activation='relu', padding='valid', use_bias=True,name="Mapping")(conv2)
    conv_trans = Conv2DTranspose(1,7,padding='same')(conv3)
    ARCNN = Model(inputs=inp,outputs=conv_trans,name="ARCNN")
    return ARCNN

def get_Fast_ARCNN(input_shape=(32,32,1)):
    inp = Input(shape=input_shape)
    conv1 = Conv2D(64,9,activation='relu', padding='same', use_bias=True,name="Feature_extract")(inp)
    conv2 = Conv2D(32,1,activation='relu', padding='valid', use_bias=True,name="Feature_Enhance_speed")(conv1)
    conv3 = Conv2D(32,7,activation='relu', padding='same', use_bias=True,name="Feature_Enhance")(conv2)
    conv4 = Conv2D(64,1,activation='relu', padding='valid', use_bias=True,name="Mapping")(conv3)
    conv_trans = Conv2DTranspose(1,7,padding='same')(conv4)
    ARCNN = Model(inputs=inp,outputs=conv_trans,name="Faster_ARCNN")
    return ARCNN

def get_ARCNN_lite(input_shape=(32,32,1)):
    inp = Input(shape=input_shape)
    conv1 = Conv2D(32,5,dilation_rate=4,activation='relu', padding='same', use_bias=True,name="Feature_extract")(inp)
    conv2 = Conv2D(32,5,dilation_rate=2,activation='relu', padding='same', use_bias=True,name="Feature_Enhance")(conv1)
    conv3 = Conv2D(32,1,activation='relu', padding='valid', use_bias=True,name="Mapping")(conv2)
    conv4 = Conv2D(1,3,dilation_rate=4,name="Upscale",padding='same')(conv3)
    ARCNN = Model(inputs=inp,outputs=conv4,name="ARCNN_lite")
    return ARCNN


class PixelAttention(tf.keras.layers.Layer):
    def __init__(self, nf,name ='PixAttention'):
        super(PixelAttention, self).__init__(name=name)
        self._name = name
        self.conv1 = Conv2D(filters=nf,kernel_size=1)
    
    def call(self,x):
        y = self.conv1(x)
        self.sig = tf.keras.activations.sigmoid(y)
        out = tf.math.multiply(x,y)
        return out

def get_ARCNN_att(input_shape=(32,32,1)):
    inp = Input(shape=input_shape)
    conv1 = Conv2D(32,5,dilation_rate=4,activation='relu', padding='same', use_bias=True,name="Feature_extract")(inp)
    conv2 = Conv2D(32,5,dilation_rate=2,activation='relu', padding='same', use_bias=True,name="Feature_Enhance")(conv1)
    pa2 = PixelAttention(32,name="PA2")(conv2)
    conv3 = Conv2D(32,1,activation='relu', padding='valid', use_bias=True,name="Mapping")(pa2)
    pa3 = PixelAttention(32,name="PA3")(conv3)
    conv4 = Conv2D(1,3,dilation_rate=4,name="Image",padding='same')(pa3)
    ARCNN = Model(inputs=inp,outputs=conv4)
    return ARCNN