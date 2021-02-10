import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Conv2DTranspose, SeparableConv2D
from dataset import create_artifact_dataset

#Define the model
def get_ARCNN_v1(input_shape=(32,32,1)):
    inp = Input(shape=input_shape)
    conv1 = Conv2D(64,9,activation='relu', padding='same', use_bias=True,name="Feature_extract")(inp)
    conv2 = Conv2D(32,1,activation='relu', padding='valid', use_bias=True,name="Feature_Enhance_speed")(conv1)
    conv3 = Conv2D(32,7,activation='relu', padding='same', use_bias=True,name="Feature_Enhance")(conv2)
    conv4 = Conv2D(64,1,activation='relu', padding='valid', use_bias=True,name="Mapping")(conv3)
    conv_trans = Conv2DTranspose(1,7,padding='same')(conv4)
    ARCNN = Model(inputs=inp,outputs=conv_trans,name="ARCNN_v1")
    return ARCNN


def get_ARCNN_v2(input_shape=(32,32,1)):
    inp = Input(shape=input_shape)
    conv1 = Conv2D(32,5,dilation_rate=4,activation='relu', padding='same', use_bias=True,name="Feature_extract")(inp)
    conv2 = Conv2D(32,1,activation='relu', padding='valid', use_bias=True,name="Feature_Enhance_speed")(conv1)
    conv3 = Conv2D(32,5,dilation_rate=2,activation='relu', padding='same', use_bias=True,name="Feature_Enhance")(conv2)
    conv4 = Conv2D(32,1,activation='relu', padding='valid', use_bias=True,name="Mapping")(conv3)
    conv_trans = Conv2DTranspose(1,3,dilation_rate=4,name="Upscale",padding='same')(conv4)
    #conv5 = Conv2D(1,5,activation='relu', padding='same', use_bias=True,name="SR_Mapping")(conv4)
    ARCNN = Model(inputs=inp,outputs=conv_trans)
    return ARCNN
#Define the metrics

def ssim(y_true,y_pred):
    return tf.image.ssim(y_true,y_pred,max_val=1.0)

def psnr(y_true,y_pred):
    return tf.image.psnr(y_true,y_pred,max_val=1.0)

@tf.function
def custom_loss(y_true, y_pred):
    alpha = tf.constant(0.84)
    mssim = alpha*(1-tf.image.ssim_multiscale(y_true,y_pred,max_val=1.0,filter_size=3))
    mse = tf.metrics.mae(y_true, y_pred)
    loss = tf.reduce_mean(mssim) + (1-alpha)*tf.reduce_mean(mse)
    return loss

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #Create Model
    ver = 2
    if(ver == 1):
        model = get_ARCNN_v1((None,None,1))
        print(model.summary())
    else:
        model = get_ARCNN_v2((None,None,1))
        model.summary()

    #Load Dataset
    data = create_artifact_dataset()
    data = data.prefetch(tf.data.experimental.AUTOTUNE)
    
    #Set callbacks
    tboard = tf.keras.callbacks.TensorBoard(log_dir="./logs/ARCNN_ssim",write_images=True)
    filepath="./checkpoints/ARCNN_ssim/weights-improvement-{epoch:02d}-{ssim:.2f}.hdf5"
    cp = tf.keras.callbacks.ModelCheckpoint(filepath,monitor="ssim",verbose=1,save_weights_only=True)
    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='ssim', factor=0.1, patience=5, verbose=1,mode='max',
                                                     min_delta=0.001, 
                                                     cooldown=2, 
                                                     min_lr=1e-6)

    #Train Model
    optim = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optim,loss=custom_loss,metrics=[ssim,psnr])
    model.fit(data,epochs=40,callbacks=[tboard,cp,lr_reduce])

    #SaveModel
    model.save("./models/ARCNN_spatch",save_format="tf")