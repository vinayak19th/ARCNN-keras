import tensorflow as tf
import numpy as np
from dataset import create_SR_dataset
import argparse


def get_args():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-p','--model_load_path',type=str,help='Path to Saved_model to load',required=True)
    my_parser.add_argument('-m','--model_save_path',type=str,help='Path to Saved_model',required=True)
    my_parser.add_argument('-c','--checkpoint_path',type=str,help='Path to checkpoints',required=True)
    my_parser.add_argument('-l','--log_path',type=str,help='Path to logdir',required=True)
    my_parser.add_argument('-e','--epochs',type=int,help='Number of epochs',default=50)
    return my_parser

@tf.function
def ssim(y_true,y_pred):
    return tf.image.ssim(y_true,y_pred,max_val=1.0)

@tf.function
def psnr(y_true,y_pred):
    return tf.image.psnr(y_true,y_pred,max_val=1.0)

@tf.function
def custom_loss(y_true, y_pred):
    alpha = tf.constant(0.84)
    mssim = alpha*(1-tf.image.ssim_multiscale(y_true,y_pred,max_val=1.0,filter_size=3))
    mse = tf.metrics.mae(y_true, y_pred)
    loss = tf.reduce_mean(mssim) + (1-alpha)*tf.reduce_mean(mse)
    return loss

def makedirs(opt):
    try:
        os.mkdir(opt.checkpoint_path)
    except:
        pass
    try:
        os.mkdir(opt.log_path)
    except:
        pass

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    opt = get_args().parse_args()
    makedirs(opt)


    SARCNN = tf.keras.models.load_model(opt.model_load_path,custom_objects={'ssim':ssim,'psnr':psnr,'custom_loss':custom_loss})
    SARCNN.trainable = True
    SARCNN.summary()
    
    tboard = tf.keras.callbacks.TensorBoard(log_dir=opt.log_path,write_images=True)
    filepath= opt.checkpoint_path+"/weights-{epoch:03d}-{ssim:.4f}.hdf5"
    cp = tf.keras.callbacks.ModelCheckpoint(filepath,monitor="ssim",verbose=1,save_weights_only=True)
    
    print("Starting Fine Tuning")
    data = create_SR_dataset(fpath="IPLData/480/",batch_size=16,p=150,s=35,jpq=(70,100),fformat="*.png")
    data = data.prefetch(tf.data.experimental.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    SARCNN.compile(optimizer=optimizer,loss=custom_loss,metrics=[ssim,psnr])
    SARCNN.fit(data,epochs=opt.epochs,callbacks=[tboard,cp])

    SARCNN.save(opt.model_save_path,save_format="tf")    