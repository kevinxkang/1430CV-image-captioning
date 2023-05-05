from flickr8k import Flickr8kDataset
from flickr30k import Flickr30kDataset
from coco120k import Coco120KDataset
import shutil
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #flickr_8kdataset = Flickr8kDataset("/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/flickr8k_images/", "/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/flickr8k_images/captions.txt")
    # flickr_30kdataset = Flickr30kDataset("/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/flickr30k_images/", "/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/flickr30k_images/results.csv")
    coco120k_imags = Coco120KDataset("/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/coco_120k/train2014/train2014/","/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/coco_120k/val2014/val2014/","/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/coco_120k/captions/annotations/captions_train2014.json", "/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/coco_120k/captions/annotations/captions_val2014.json", max_samples=30000)
    
    batch_images, batch_unprocess_captions, batch_processed_captions = coco120k_imags[0]
    plt.imshow(batch_images[0].astype("uint8"))
    plt.show()
    print("Captions for the first image:")
    print(batch_unprocess_captions[0])
    print(batch_processed_captions[0])