from nnunetv2.run.run_training import run_training
import argparse 
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default="Dataset_Train_val",
                        help="[REQUIRED] the training data's name ")
    parser.add_argument('-device', type=str, default='cpu', required=False,
                    help="Use this to set the device the training should run with. Available options are 'cuda' "
                         "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                         "Use CUDA_VISIBLE_DEVICES=X python train.py [...] instead!")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
                        
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    
    args = parser.parse_args()

    
    print('Training...')
    run_training(args.d, configuration='2d', fold = args.fold, pretrained_weights = args.pretrained_weights, device = torch.device(args.device))
    