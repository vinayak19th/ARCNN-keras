import tensorflow as tf
import numpy as np
from dataset import create_artifact_dataset
import model

#Define opt
def get_args():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-m','--model_save_path',type=str,help='Path to Saved_model',required=True)
    my_parser.add_argument('-c','--checkpoint_path',type=str,help='Path to checkpoints',required=True)
    my_parser.add_argument('-l','--log_path',type=str,help='Path to logdir',required=True)
    my_parser.add_argument('-v','--version',type=int,help='ARCNN version to train 1: Original | 2: Fast ARCNN | 3: Dilated |4. Attention',required=True,choices=[1,2,3,4])
    my_parser.add_argument('-e','--epochs',type=int,help='Number of epochs',default=50)
    my_parser.add_argument('-d','--dataset',type=str,help='Path to folder of images for training',required=True)
    
    #Optional Args
    my_parser.add_argument('--batch_size',type=int,help='Batch size',default=16)
    my_parser.add_argument('--patch_size',type=int,help='Patch size for training',default=100)
    my_parser.add_argument('--stride_size',type=int,help='Stride of patches',default=35)
    my_parser.add_argument('--jpq_upper',type=int,help='Highest JPEG quality for compression',default=20)
    my_parser.add_argument('--jpq_lower',type=int,help='Lowest JPEG quality for compression',default=10)
    return my_parser


#Define the metrics

def ssim(y_true,y_pred):
    return tf.image.ssim(y_true,y_pred,max_val=1.0)

def psnr(y_true,y_pred):
    return tf.image.psnr(y_true,y_pred,max_val=1.0)

@tf.function
def custom_loss(y_true, y_pred):
    alpha = tf.constant(0.5)
    mssim = alpha*(1-tf.image.ssim_multiscale(y_true,y_pred,max_val=1.0,filter_size=3))
    mse = tf.metrics.mae(y_true, y_pred)
    loss = tf.reduce_mean(mssim) + (1-alpha)*tf.reduce_mean(mse)
    return loss

#Create Dirs
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
    for i in physical_devices:
        tf.config.experimental.set_memory_growth(i, True)
    # Get args
    opt = get_args().parse_args()
    #make dirs
    makedirs(opt)
    #Create Model
    ver = 2
    if (opt.version == 1):
        model = model.get_ARCNN((None,None,1))
    elif (opt.version == 2):
        model = model.get_Fast_ARCNN((None,None,1))
    elif (opt.version == 3):
        model = model.get_ARCNN_lite((None,None,1))
    elif (opt.version == 4):
        model = model.get_ARCNN_att((None,None,1))

    #Load Dataset
    data = create_artifact_dataset(fpath=opt.dataset,
        batch_size=opt.batch_size,
        p=opt.patch_size,
        s=opt.stride_size,
        jpq=(opt.jpq_lower,opt.jpq_upper)))
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
    model.save(opt.model_save_path,save_format="tf")