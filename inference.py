import argparse
import os
import torch

from utils.util import *

from pre_process.preprocess import preprocess
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data as predict
from post_process.remove_small_segments import remove_small_segments

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, required=False, default='./model_folder', help = 'model folder')
    parser.add_argument('-chk', type=str, required=True, default='./model_folder/model_final.pth',help = 'checkpoint name, put in the model folder')
    
    parser.add_argument('-i', type=str, required=False, default='./dataset_test/raw/', help = 'Folder in which the raw test images are')
    parser.add_argument('-o', type=str, required=False, default='./dataset_test/raw_prediction/', help = 'Folder in which the raw predictions are')
    parser.add_argument('-p', type=str, required=False, default='./dataset_test/post_prediction/', help = 'Folder in which the post predictions are')
    
    parser.add_argument('-t', type=int, required=False, default=600, help = 'Threshold to remove small segments')
 
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('--------------preprocessing--------------')

    preprocess_path = './dataset_test/preprocessed/'
    mkdir(preprocess_path)
    
    for file_name in os.listdir(args.i):
        
        img_array = image_to_array(os.path.join(args.i, file_name))
        pre_img = preprocess(img_array)
        cv2.imwrite(os.path.join (preprocess_path, file_name), pre_img)
    print('--------------preprocessing done--------------')
    mkdir(args.o)
    print('--------------predicting--------------')
    predict(list_of_lists_or_source_folder = preprocess_path, output_folder = args.o, model_training_output_dir = args.m, use_folds =[0], checkpoint_name=args.chk, num_processes_preprocessing=1, num_processes_segmentation_export=1,device = device)
    print('--------------postprocessing--------------')
    mkdir(args.p)
    remove_small_segments(args.o, args.p, threshold = args.t)
    print('--------------Done--------------')
    