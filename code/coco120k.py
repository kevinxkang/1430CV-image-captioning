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
import json
from multiprocessing import Pool

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

def load_image_external(image_id, image_paths, preprocessing_steps, r_image_height, r_image_width):
    image_path = image_paths[image_id]
    try:
        img = imread(image_path)
        if 'resize' in preprocessing_steps:
            img = cv2.resize(img, (r_image_height, r_image_width))
        return img
    except Exception as e:
        print(e)
        print(f"Was unable to load image {image_id}")
        return None
"""
In order to use keras, inhereting from a sequence allows us to take advantage of multiprocessing for large datasets. This will help for bigger data as well. Using a DataGenerator class speeds process up
"""
class Coco120KDataset(Sequence):
    def __init__(self, train_image_dir, val_image_dir, train_caption_file, val_caption_file, batch_size=32, preprocessing=['resize', 'normalize', 'augment'], combine=True, max_samples=None):
        self.train_image_dir = train_image_dir
        self.val_image_dir = val_image_dir
        self.preprocessing_steps = preprocessing
        self.r_image_height = 224
        self.r_image_width = 224
        self.max_samples = max_samples
        self.combine = combine
        self.index_dict = {}

        self.captions, self.image_paths = self.load_captions_images(train_caption_file, val_caption_file)
        self.image_ids = list(self.captions.keys())
        self.batch_size = batch_size

        # Preprocessing and augmentation parameters
        self.width_shift_range = 0.1
        self.height_shift_range = 0.1
        self.brightness_range = (0.8, 1.2)
        self.rotation_range = 20
        self.horizontal_flip = True

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
        
        self.images = self.load_images(self.image_paths)
            
    def load_captions_images(self, train_caption_file, val_caption_file):
        print("Loading captions and images")
        # Helper function to load the captions and images from the JSON files
        def load_caption_image_data(caption_file, image_dir):
            with open(caption_file, 'r') as f:
                data = json.load(f)
            
                annotations = data['annotations']
                images_data = data['images']
                
                
                    
                captions = {}
                images = {}

                for img_data in images_data:
                    img_id = str(img_data['id'])
                    file_name = img_data['file_name']
                    images[img_id] = os.path.join(image_dir, file_name)

                for annotation in annotations:
                    image_id = str(annotation['image_id'])
                    caption_id = str(annotation['id'])
                    caption = annotation['caption']

                    if image_id not in captions:
                        captions[image_id] = []
                    captions[image_id].append(caption)

                return captions, images

        # Load train and validation captions and images
        train_captions, train_images = load_caption_image_data(train_caption_file, self.train_image_dir)
        val_captions, val_images = load_caption_image_data(val_caption_file, self.val_image_dir)

        if self.combine:
            # Combine train and validation data
            all_captions = {**train_captions, **val_captions}
            all_images = {**train_images, **val_images}
            return all_captions, all_images
        else:
            return (train_captions, val_captions), (train_images, val_images)
                
        

    """
    Purpose: This function processes the captions with various NLP tactics 
    Input: Caption -> str --> Caption for a photo
    Output: Caption --> arr[str] --> Processed captions 
    """
    def process_caption(self, caption):
        print("Processing captions")
        processed_captions = []
        for cap in caption:
            doc = nlp(cap.lower())
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop and not token.is_punct and not token.like_num
            ]
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
        if len(image.shape) == 3:  # RGB image
            return (image - self.mean) / self.std
        elif len(image.shape) == 2:  # Grayscale image
            return (image - np.mean(self.mean)) / np.mean(self.std)
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
    Purpose: To extract all the images from the dataset
    Input: image_dir --> str --> representing the image directory
    Output: processed_images -> arr[images] --> returns an array of images
    """
    def load_images(self, image_paths):
        print("Loading images")

        def load_image(image_id):
            image_path = image_paths[image_id]
            try:
                img = imread(image_path)
                if 'resize' in self.preprocessing_steps:
                    img = self.resize_image(img)
                return img
            except Exception as e:
                print(e)
                print(f"Was unable to load image {image_id}")
                return None
        
        

        with Pool(processes=2) as pool:
            loaded_images = list(tqdm(pool.starmap(load_image_external, [(image_id, self.image_paths, self.preprocessing_steps, self.r_image_height, self.r_image_width) for image_id in self.captions.keys()]), total=len(self.captions)))

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
    # Must implement for the SequenceClass
    # GetItem returns a batch of images and captions. 
    # Input: idx -- of batch
    # Output: images, captions --> tuple of batched images, captions
    # """
    def __getitem__(self, idx):
        batch_image_ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_indices = [self.index_dict[image_id] for image_id in batch_image_ids if image_id in self.index_dict]
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