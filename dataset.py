import tensorflow as tf
import glob
import numpy as np
import os

#Dataset Preprocessing Function for artifact dataset

@tf.function
def create_pairs(flist,jpq=(10,20)):
    images = tf.TensorArray(tf.float16,dynamic_size=True,size=0,infer_shape=True)
    images_comp = tf.TensorArray(tf.float16,dynamic_size=True,size=0,infer_shape=True)
    c =0
    for file in flist:
        y = tf.image.decode_jpeg(tf.io.read_file(file))
        x = tf.image.random_jpeg_quality(y,jpq[0],jpq[1])
        y = tf.expand_dims(tf.image.rgb_to_yuv(tf.cast(y,tf.float16))[:,:,0],-1)/255
        x = tf.expand_dims(tf.image.rgb_to_yuv(tf.cast(x,tf.float16))[:,:,0],-1)/255
        images = images.write(c,y)
        images_comp =images_comp.write(c,x)
        c+=1
    y = images.stack()
    x = images_comp.stack()
    return (x,y)

@tf.function
def create_patches(x,y,p,s):
    print("Shapes")
    batch_size = tf.shape(y)[0]
    print(batch_size)
    #Extracting patches and converting into batches
    y_patches = tf.image.extract_patches(images=y,sizes=(1,p,p,1),strides=(1,s,s,1),rates=(1,1,1,1),padding='VALID')
    #Calculating patch sizes and batches
    shapes= tf.shape(y_patches)
    patch_batch = int(shapes[1]*shapes[2]*batch_size)
    
    y_patches = tf.reshape(y_patches,(patch_batch,p,p,1))
    print("y_patches :",y_patches.shape)
    
    x_patches = tf.image.extract_patches(images=x,sizes=(1,p,p,1),strides=(1,s,s,1),rates=(1,1,1,1),padding='VALID')
    x_patches = tf.reshape(x_patches,(patch_batch,p,p,1))
    print("x_patches :",x_patches.shape)
    return (x_patches,y_patches)

#Dataset Wrapper
def create_artifact_dataset(fpath = "",batch_size=32,p=200,s=42,jpq=(10,20),fformat="jpg"):
    """
    Wrapper function to return tf.dataset object with all the data
        fpath : Path to folder containing jpeg files
            ex:HarmonicI_720p_1000k_1440p_bicubic/480/
            HR should be a similar directory with the parent changed from 480 to 960
            ex:HarmonicI_720p_1000k_1440p_bicubic/960/
        batch_size : size of batches per batch of patches
        p : Patch size
        s : stride size
        jpq : Tuple(min,max)
            ex: jpq = (10,20) ; where min quality is 10 and max is 20
    """
    flist = glob.glob(os.path.join(fpath,"*."+fformat))
    print("flist:",len(flist))
    artifact_dataset = tf.data.Dataset.from_tensor_slices(flist).batch(32)
    
    func = lambda x: create_pairs(x,jpq)
    artifact_dataset = artifact_dataset.map(func,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("JPEG Pairs created with quality of range:",jpq,"\n--------------------")
    
    func = lambda x,y: create_patches(x,y,p,s)
    artifact_dataset = artifact_dataset.map(func,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("Created Patches\n--------------------")
    
    artifact_dataset = artifact_dataset.unbatch().batch(batch_size)
    print("Dataset batches of batch size",batch_size,"\n--------------------")
    print("Dataset Spec:\n",artifact_dataset.element_spec)
    
    artifact_dataset = artifact_dataset.cache()
    return artifact_dataset

