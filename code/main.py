import os
import argparse
import pickle
import numpy as np
from  architecture.model import ImageCaptionModel

from datasets.preprocess import DatasetProcessor

def parse_args(args=None):
    """
    Command line argument parser
    """

    parser = argparse.ArgumentParser(description="JELLYFAM's Image Captioning Project", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #Avalible arguments
    parser.add_argument("--dataset", required=True, type=str, default="flickr8k", help="Dataset to be used for training")
    parser.add_argument("--task", required=True, type=str, choices=["train", "test", "both"], default="train", help="Task to be performed")
    parser.add_argument("--processed", required=True, type=bool, default=False, help="If data has already been pre-processed")
    # parser.add_argument("--decoder", required=True, type=str, default="lstm", help="Decoder to be used for training")
    # parser.add_argument('--epochs',         type=int,   default=3,      help='Number of epochs used in training.')
    # parser.add_argument('--lr',             type=float, default=1e-3,   help='Model\'s learning rate')

    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.


def main(args):

    # Load Data
    if args.processed:
        with open(f'../data/{args.dataset}/{args.dataset}.p', 'rb') as data_file:
            data_dict = pickle.load(data_file)
    else:
        dataset = DatasetProcessor(args.dataset)
        data_dict = dataset.data_dict

    train_captions = data_dict["train_captions"]
    test_captions = data_dict["test_captions"]
    train_image_features = data_dict["train_image_features"]
    test_image_features = data_dict["test_image_features"]
    word2idx = data_dict["word2idx"]
    idx2word = data_dict["idx2word"]

    if args.task in ["train", "both"]:
        # Train Model
        decoder = {
           #TODO 
        }

        print("Training Model...")
        # print(train_image_features)
        model = ImageCaptionModel(decoder)
        

        

if __name__ == '__main__':
    main(parse_args())






    