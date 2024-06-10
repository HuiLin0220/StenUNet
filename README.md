<a href="https://arxiv.org/abs/2310.14961" alt="Citation"><img src="https://img.shields.io/badge/cite-citation-blue" /></a>
# Introduction
This algorithm is for the stenosis detection task in [ARCADE Challenge](https://arcade.grand-challenge.org/), which was held at MICCAI 2023. We are ranked ${\textsf{\color{red}3rd}}$ !

Our publication:  StenUNet: Automatic Stenosis Detection from X-ray Coronary Angiography [Arxiv](https://arxiv.org/abs/2310.14961)

Please refer to [MICCAI-ARCADE](https://github.com/NMHeartAI/MICCAI_ARCADE.git) for the segmentation detection task.


## Installation
python>=3.9 and torch>=2.0.0

      conda create -n stenunet_env python=3.9
      conda activate stenunet_env
      git clone https://github.com/HuiLin0220/StenUNet.git
      cd StenUNet
      pip install  -r ./requirements.txt

## Prepare data

1. Rename and put the training images in this folder "./nnNet_training/Raw_data/"
Training folder structure is like this:

   Raw_data/Dataset_train_val/  
          ├── raw
          │   ├── sten_0000_0000.png
          │   ├── sten_0000_0001.png
          │   ├── ...
          │   ├── sten_0001_0000.png      
          │   ├── sten_0001_0001.png      
          │   ├── ... 
          │   ├── sten_0002_0000.png
          │   ├── sten_0002_0001.png
          │   ├── ...
          ├── labelsTr
          │   ├── sten_0000.png
          │   ├── sten_0001.png
          │   ├── sten_0002.png
          │   ├── ...
## Train
- StenUnet's weight ([Google drive](https://drive.google.com/file/d/1BO4whry0i50h_yzqQwUw1k7QyyLUk2U3/view?usp=sharing)).
## Inference
1. Rename and put the test images in this folder'./dataset_test/raw';
2. Run
  
         python evaluation.py -chk MODEL_WEIGHTS_PATH;
   
5. You will get the preprocessed images, raw prediction after StenUNet, and post_prediction after postprocessing.

You can integrate your own preprocessing/postprocessing strategies in [preprocess.py](pre_process/preprocess.py)/[post_process](post_process/remove_small_segments.py)

The inference folder structure is like this:

      daset_test/
          ├── raw
          │   ├── sten_0000_0000.png
          │   ├── sten_0001_0000.png
          │   ├── ...
          ├── preprocessed
          │   ├── sten_0000_0000.png       # prerpocessing method0
          │   ├── sten_0000_0001.png       # prerpocessing method1
          │   ├── sten_0000_0003.png       # prerpocessing method2
          │   ├── ... 
          │   ├── sten_0001_0000.png
          │   ├── sten_0001_0001.png
          │   ├── sten_0001_0003.png
          │   ├── ...
          ├── raw_prediction
          │   ├── sten_0000.png
          │   ├── sten_0001.png
          │   ├── ...
          ├── post_prediction
          │   ├── sten_0000.png
          │   ├── sten_0001.png
          │   ├── ...
## References
[nnunet](https://github.com/MIC-DKFZ/nnUNet)

## Citation
Please cite the following paper when using SteUNet:

      @article{lin2023stenunet,
        title={StenUNet: Automatic Stenosis Detection from X-ray Coronary Angiography},
        author={Lin, Hui and Liu, Tom and Katsaggelos, Aggelos and Kline, Adrienne},
        journal={arXiv preprint arXiv:2310.14961},
        year={2023}
      }

## Contact Us
Feel free to contact me at huilin2023@u.northwestern.edu

## To-do List
