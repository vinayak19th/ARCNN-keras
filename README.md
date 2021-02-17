# ARCNN-keras
A tf-keras implementation of ARCNN mentioned in :
* [Deep Convolution Networks for Compression Artifacts Reduction](https://arxiv.org/abs/1608.02778), Ke Yu, Chao Dong, Chen Change Loy, Xiaoou Tang



## Requirnments:
* Tensorflow
* tqdm
* Numpy
* Pillow
* Everything within the [requirements.txt](./requirements.txt) file.

> Dockerfile with all required libs included in repository

## Dataset:
The scripts are written to be trained on any folder of images as long as:
1. All images are of the same dimensions
2. All images are of the same file format
> example : A folder where all images are pngs and 720p (1280x720)

*Recommended that you use [Div2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/)*, Use HR or LR based on the closest match to target inference domain.

## Models
There are 3 seperate models for training: The ARCNN, Faster ARCNN, ARCNN Lite (A faster ARCNN with dilated convolutions)

The comparision in parameters is given below:
<table border=2>
    <tr>
        <td><b>Model</b></td>
        <td><b>Paramenters</b></td>
    </tr>
    <tr>
        <td>ARCNN</td>
        <td>108k</td>
    </tr>
    <tr>
        <td>Faster ARCNN</td>
        <td>64k</td>
    </tr>
    <tr>
        <td>ARCNN Lite</td>
        <td>32k</td>
    </tr>
</table>
<br>

## Sample Results

### Ground Truth
<center><img src="./test/GT/butterfly.jpg" height="300"></center>

> All outputs are from the dilated model 

<center>
<table>
    <tr>
        <td><center>JPEG 20</center></td>
        <td><center>Inference</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./test/Inputs/butterfly20.jpg" height="150"></center>
    	</td>
    	<td>
    		<center><img src="./test/outputs/outbutterfly20.jpg" height="150"></center>
    	</td>
    </tr>
    <tr>
        <td><center>JPEG 15</center></td>
        <td><center>Inference</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./test/Inputs/butterfly15.jpg" height="150"></center>
    	</td>
    	<td>
    		<center><img src="./test/outputs/outbutterfly15.jpg" height="150"></center>
    	</td>
    </tr>
    <tr>
        <td><center>JPEG 10</center></td>
        <td><center>Inference</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./test/Inputs/butterfly10.jpg" height="150"></center>
    	</td>
    	<td>
    		<center><img src="./test/outputs/outbutterfly10.jpg" height="150"></center>
    	</td>
    </tr>
</table>
</center>


# Usage

## Docker 
The docker container includes all the packages plus jupyter lab for ease of use.
**Remember to pass the flag "--ip 0.0.0.0" to jupyter lab**

The usage of docker would be the following:

### _Step 1: Enter repo folder_
```bash
> cd /location/of/repository
```
### _Step 2: Build image from Dockerfile
```bash
docker build ./ -t arcnnkeras  
```
### _Step 3: Start and enter container
```bash
docker run -it --gpus all -v $PWD:/app -p 8888:8888 arcnnkeras bash
```
Notes:
* Use the "--gpus" flag only is [nvidia container runtime](https://github.com/NVIDIA/nvidia-container-runtime) is set up
* -v parameters can be added for different data folders that need to be mounted
* The port 8888 is passed for jupyter usage. It isn't needed for inference

## Inference

Inside the docker container or in your env use the [infer.py](./infer.py) script to infer the results on a folder of images.

***Folder should follow the rules***
1. All images are of the same dimensions
2. All images are of the same file format

```bash
usage: infer.py [-h] -p FOLDER_PATH -m MODEL_PATH -o OUTPUT_PATH

optional arguments:
  -h, --help            show this help message and exit
  -p FOLDER_PATH, --folder_path FOLDER_PATH
                        Path to folder of frames
  -m MODEL_PATH, --model_path MODEL_PATH
                        Path to weights file
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to output folder
```

***Example***
```bash
python infer.py -m ./models/Model2_dialted_epoch/ -p ./test/Inputs/ -o ./test/
```

* Pre-Trained models are found in the model folder. Link one of them in the -m command. *Reccomeneded* Model2_dialted_epoch

## Training
Use the [train_ARCNN.py](./train_ARCNN.py) script in order to train the ARCNN model.

```bash
usage: train_ARCNN.py [-h] -m MODEL_SAVE_PATH -c CHECKPOINT_PATH -l
                               LOG_PATH -d DATASET_PATH [-f FILE_FORMAT] -v
                               {1,2,3} [-e EPOCHS] [--batch_size BATCH_SIZE]
                               [--patch_size PATCH_SIZE]
                               [--stride_size STRIDE_SIZE]
                               [--jpq_upper JPQ_UPPER] [--jpq_lower JPQ_LOWER]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_SAVE_PATH, --model_save_path MODEL_SAVE_PATH
                        Path to Saved_model
  -c CHECKPOINT_PATH, --checkpoint_path CHECKPOINT_PATH
                        Path to checkpoints
  -l LOG_PATH, --log_path LOG_PATH
                        Path to logdir
  -d DATASET_PATH, --dataset_path DATASET_PATH
                        Path to Folder of images
  -f FILE_FORMAT, --file_format FILE_FORMAT
                        Format of images
  -v {1,2,3}, --version {1,2,3}
                        ARCNN version to train 1: Original | 2: Fast ARCNN |
                        3: Dilated
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  --batch_size BATCH_SIZE
                        Batch size
  --patch_size PATCH_SIZE
                        Patch size for training
  --stride_size STRIDE_SIZE
                        Stride of patches
  --jpq_upper JPQ_UPPER
                        Highest JPEG quality for compression
  --jpq_lower JPQ_LOWER
                        Lowest JPEG quality for compression
```
