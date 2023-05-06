from keras.utils import Sequence 
import os 
import pandas as pd
from PIL import Image
import cv2
from skimage.io import imread
import numpy as np
import spacy
import numpy as np
from scipy import ndimage
import random
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
"""
In order to use keras, inhereting from a sequence allows us to take advantage of multiprocessing for large datasets. This will help for bigger data as well. Using a DataGenerator class speeds process up
"""
class Flickr8kDataset(Sequence):
    """
    Constructor for the dataset 
    Inputs 
    image_dir: (str) Directory for the images for the Dataset
    captions_file (str) Filename of the file with captions
    batch_size: (Optional[int]): Batch size used for batching of the dataset. 32 is the default value 
    """
    def __init__(self, image_dir, captions_file, batch_size=32, preprocessing=['resize', 'normalize', 'augment'], tokenize=True):
        self.image_dir = image_dir
        self.image_dict = None
        self.mean = None 
        self.std = None
        self.preprocessing_steps = preprocessing
        self.r_image_height = 224
        self.r_image_width = 224
        self.tokenize = tokenize
        
        
        self.captions = self.load_captions(captions_file)
        self.images = self.load_images(image_dir)
        self.image_ids = list(self.captions.keys())
        self.batch_size = batch_size 
        
        self.width_shift_range = 0.1
        self.height_shift_range = 0.1
        self.brightness_range = (0.8, 1.2)
        self.rotation_range = 20
        self.horizontal_flip = True
        
        self.augment = False
        
        
        if 'augment' in preprocessing:
            self.augment = True
            self.data_generator = ImageDataGenerator(
                rotation_range=self.rotation_range,
                width_shift_range=self.width_shift_range,
                height_shift_range=self.height_shift_range,
                brightness_range=self.brightness_range,
                horizontal_flip=self.horizontal_flip,
                fill_mode='nearest'
            )
        else:
            self.data_generator = ImageDataGenerator()
        
    """
    Purpose: This function processes the captions with various NLP tactics 
    Input: Caption -> str --> Caption for a photo
    Output: Caption --> arr[str] --> Processed captions 
    """
    def process_caption(self, caption):
        processed_captions = []
        for cap in caption:
            doc = nlp(cap.lower())
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop and not token.is_punct and not token.like_num
            ]
            if not self.tokenize:
                tokens = ' '.join(tokens)
            processed_captions.append(tokens)
        return processed_captions

    """
    Helper function to convert an image to greyscale
    Input: image --> An image to convert to grayscale
    Output: image --> grayscale version of the image
    """
    def convert_to_grey(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    """
    Helper function to normalize image
    Input: image --> pre-normalized image
    Output: image --> normalized image
    """
    def normalize_image(self, image):
        return (image - self.mean) / self.std
    """
    Helper function resize image:
    Input: image 
    """
    def resize_image(self, image):
        return cv2.resize(image, (self.r_image_height, self.r_image_width))
    """
    Helper function to compute the mean and standard deviation of the image samples
    Input: images --> array of an images 
    Output: None --> calculates mean/std for the dataset
    """
    def compute_mean_std(self, images):
        num_images = len(images)
        mean_sum = np.zeros(3)
        std_sum = np.zeros(3)

        for image in images:
            if len(image.shape) == 3:  # RGB image
                mean_sum += np.mean(image, axis=(0, 1))
                std_sum += np.std(image, axis=(0, 1))
            elif len(image.shape) == 2:  # Grayscale image
                mean_sum += np.mean(image)
                std_sum += np.std(image)


        self.mean = mean_sum / num_images
        self.std = std_sum / num_images
    
    """
    Purpose: To extract the captions from the caption file and return them as an index dictionary with photo_id and captions
    Input: caption_file --> str --> representing the file with the captions
    Output: captions --> dict[str] --> Dictionary mapping photos to 5 captions
    """
    def load_captions(self, caption_file):
        print("Loading captions")

        def process_line(line):
            image_file, caption = line.split(',', 1)
            # processed_caption = self.process_caption(caption)
            return image_file, caption

        with open(caption_file, 'r') as f:
            content = f.readlines()

        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_line, content[1:]), total=len(content) - 1))

        image_captions = {}
        for image_file, processed_caption in results:
            if image_file not in image_captions:
                image_captions[image_file] = []
            image_captions[image_file].append(processed_caption)

        return image_captions

    """
    Purpose: To extract all the images from the dataset
    Input: image_dir --> str --> representing the image directory
    Output: processed_images -> arr[images] --> returns an array of images
    """
    def load_images(self, image_dir):
        print("Loading images")

        def load_image(image_id):
            image_path = os.path.join(image_dir, image_id)
            try:
                img = imread(image_path)
                if 'resize' in self.preprocessing_steps:
                    img = self.resize_image(img)
                return img
            except Exception as e:
                print(e)
                print(f"Was unable to load image {image_id}")
                return None

        with ThreadPoolExecutor() as executor:
            loaded_images = list(tqdm(executor.map(load_image, self.captions.keys()), total=len(self.captions)))

        loaded_images_with_indices = [(img, i) for i, (img, image_id) in enumerate(zip(loaded_images, self.captions.keys())) if img is not None]

        loaded_images, indices = zip(*loaded_images_with_indices)
        self.index_dict = {image_id: i for i, image_id in zip(indices, self.captions.keys()) if i in indices}

        self.compute_mean_std(loaded_images)
        
        processed_images = [self.normalize_image(img) if 'normalize' in self.preprocessing_steps else img for img in loaded_images]

        return processed_images
        
    """
    Must implement for the SequenceClass
    Returns number of batches in the dataset
    Input: None
    Output: num_batches --> int: represents number of batches in dataset
    """
    def __len__(self):
        return (len(self.image_ids) + self.batch_size - 1) // self.batch_size
    """
    Must implement for the SequenceClass
    GetItem returns a batch of images and captions. 
    Input: idx -- of batch
    Output: images, captions --> tuple of batched images, captions
    """
    def __getitem__(self, idx):
        batch_image_ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_indices = [self.index_dict[image_id] for image_id in batch_image_ids]
        batch_images = [self.images[i] for i in batch_indices]
        batch_unprocessed_captions = [self.captions[image_id] for image_id in batch_image_ids]
        
        
        batch_captions = [self.process_caption(caption) for caption in batch_unprocessed_captions]
        
        print("Augmenting")
        if self.augment:
            batch_images = np.array(batch_images)
            augmented_images = []
            for image in batch_images:
                # Reshape the image to add the batch dimension (required by ImageDataGenerator)
                image = image[np.newaxis]
                # Generate a single augmented image
                augmented_image = next(self.data_generator.flow(image, batch_size=1))
                # Remove the batch dimension
                augmented_image = np.squeeze(augmented_image, axis=0)
                augmented_images.append(augmented_image)
            batch_images = np.array(augmented_images)
            
        return np.array(batch_images), np.array(batch_unprocessed_captions), np.array(batch_captions)

