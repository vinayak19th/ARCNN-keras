import tensorflow as tf
import numpy as np
import PIL
import glob
import os
import argparse

def get_args():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-p','--folder_path',type=str,help='Path to folder of frames',required=True)
    my_parser.add_argument('-m','--model_path',type=str,help='Path to weights file',required=True)
    my_parser.add_argument('-o','--output_path',type=str,help='Path to output folder',required=True)
    return my_parser


def ssim(y_true,y_pred):
    return tf.image.ssim(y_true,y_pred,max_val=1.0)

def psnr(y_true,y_pred):
    return tf.image.psnr(y_true,y_pred,max_val=1.0)

def process_image_SR(impath):
    im = PIL.Image.open(impath)
    im = im.convert('YCbCr') # For single channel inference
    im = np.asanyarray(im)
    y = np.expand_dims(im[:,:,0],-1)/255 # Normalizing input
    uv = np.asanyarray(im)[:,:,1:]
    #print("uv:",uv.shape,"| y:",y.shape)
    return (y,uv)

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for i in physical_devices:
        tf.config.experimental.set_memory_growth(i, True)
    opt = get_args().parse_args()
    try:
        os.mkdir(opt.output_path)
    except:
        pass
    ARCNN = tf.keras.models.load_model(opt.model_path,custom_objects={"ssim":ssim,"psnr":psnr})
    print("Looking at folder",os.path.join(opt.folder_path,"*"))
    flist = np.asarray(glob.glob(os.path.join(opt.folder_path,"*")))
    count = 0
    total = len(flist)
    print("Processing",total,"files")
    prog = tf.keras.utils.Progbar(total,unit_name='frames')
    div = len(flist)/8
    rem = (len(flist)%8) *-1
    print("rem:",rem)
    if(rem==0):
        rem_files = []
        flist = flist.reshape(int(len(flist)/8),8)
    else:
        rem_files = flist[rem:]
        flist = flist[:rem].reshape(int(len(flist)/8),8)    
    print("Batched Files:",len(flist)*4,"| rem =",len(rem_files))
    for i in flist:
        im_y = []
        im_uv = []
        for j in range(8):
            y,uv = process_image_SR(i[j])
            im_y.append(y)
            im_uv.append(uv)
        im_y = np.stack(im_y,axis=0)
        #print(im_y.shape)
        outs = ARCNN.predict(im_y)
        for y,uv,j in zip(outs,im_uv,range(8)):
            count += 1
            out = y.reshape(im_y.shape[1], im_y.shape[2])
            y_pred = np.stack([out*255,uv[:,:,0],uv[:,:,1]],axis=-1)
            y_pred= np.clip(y_pred,0,255).astype('uint8')
            y_pred = PIL.Image.fromarray(y_pred,mode='YCbCr').convert('RGB')
            fname = "out"+ i[j].split("/")[-1]
            converter = PIL.ImageEnhance.Color(y_pred)
            y_pred = converter.enhance(1.4)
            y_pred.save(opt.output_path+fname)
            prog.update(count)
        #print("=",end="")     
    print(count,"Files done")
    for i in rem_files:
        im_y,im_uv = process_image_SR(i)
        #print(im_y.shape)
        im_y = np.expand_dims(im_y,0)
        outs = ARCNN.predict(im_y)
        count += 1
        out = outs.reshape(im_y.shape[1], im_y.shape[2]) #Removing batch dimensions
        y_pred = np.stack([out*255,im_uv[:,:,0],im_uv[:,:,1]],axis=-1)
        y_pred= np.clip(y_pred,0,255).astype('uint8')
        y_pred = PIL.Image.fromarray(y_pred,mode='YCbCr').convert('RGB')
        fname = "out"+ i.split("/")[-1]
        converter = PIL.ImageEnhance.Color(y_pred)
        y_pred = converter.enhance(1.4)
        y_pred.save(opt.output_path+fname)
        prog.update(count)
    print("\nDone")
