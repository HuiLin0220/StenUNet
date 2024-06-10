from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints,plan_experiments,preprocess_dataset

import argparse 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default="Dataset_Train_val",
                        help="[REQUIRED] the training data's name ")
    
    args = parser.parse_args()
    print("Fingerprint extraction...")
    extract_fingerprints(args.d, check_dataset_integrity=True)
    
    print('Experiment planning...')
    plan_experiments(args.d)
    
    print('Preprocessing...')
    preprocess_dataset(args.d, configurations=('2d',),num_processes=(8,))
    
   