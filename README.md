<a href="https://arxiv.org/abs/2310.14961" alt="Citation"><img src="https://img.shields.io/badge/cite-citation-blue" /></a>
# Introduction
This algorithm is for the stenosis detection task in [ARCADE Challenge](https://arcade.grand-challenge.org/), which was held at MICCAI 2023. We are ranked **3rd**!

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

## Train
- StenUnet's weight ([Google drive](https://drive.google.com/file/d/1BO4whry0i50h_yzqQwUw1k7QyyLUk2U3/view?usp=sharing)).
## Inference

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
