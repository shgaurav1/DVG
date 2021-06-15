# Diverse Video Generation
This is code for the paper [Diverse Video Generation using a Gaussian Process Trigger](https://openreview.net/forum?id=Qm7R_SdqTpT) by [Gaurav Shrivastava](https://www.cs.umd.edu/~gauravsh/) and [Abhinav Shrivastava](https://www.cs.umd.edu/~abhinav/). 

## Environment
1. Ensure you have [anaconda](https://www.anaconda.com/download/) installed and run:
    ```bash
    conda env create -f environment.yml # Creates environment
    conda activate dvg                  # Activates environment
    ```
2. Install torch following the instructions here [torch.ch/docs/getting-started.html](torch.ch/docs/getting-started.html). This will make it possible to run lua scripts in this repository.

3. Install [ffmpeg](https://ffmpeg.org/), this is needed for converting .avi file into .png files for the data loader.


## KTH action dataset
```bash
# Download the dataset
sh data/download_kth.sh /my/kth/data/path/

# Convert .avi to .png files
th data/convert_kth.lua --imageSize 64 --dataRoot /my/kth/data/path/

# Create metadata
python data/meta_creator.py
```

## Training the model
Experimental results in the paper used 64x64.
To train the DVG model on 64x64 KTH videos run:
```
python DVG.py 
```
<p align="center">
<img src='Results/QualitativeResults_DVG.gif' align="center" width=630 height="480">
</p>