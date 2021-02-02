# Diverse Video Generation
This is code for the paper [Diverse Video Generation using a Gaussian Process Trigger](https://openreview.net/forum?id=Qm7R_SdqTpT) by Gaurav Shrivastava(https://www.cs.umd.edu/~gauravsh/) and Abhinav Shrivastava(https://www.cs.umd.edu/~abhinav/). 



## KTH action dataset
First download the KTH action recognition dataset by running:
```
sh data/download_kth.sh /my/kth/data/path/
```
where /my/kth/data/path/ is the directory the data will be downloaded into. Next, convert the downloaded .avi files into .png's for the data loader. To do this you'll want [ffmpeg](https://ffmpeg.org/) installed. The following script will do the conversion, but beware, it's written in lua (sorry!):
```
python data/convert_lua_pytorch.py
```
The ```--imageSize``` flag specifiec the image resolution. Experimental results in the paper used 128x128, but you can also train a model on 64x64 and it will train much faster.
To train the SVG-FP model on 64x64 KTH videos run:
```
python DVG.py 
```