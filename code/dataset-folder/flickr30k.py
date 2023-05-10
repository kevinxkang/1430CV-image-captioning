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
from tensorflow.keras.layers import GlobalAveragePooling2D
import random
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.models import Model
import pickle

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
class Flickr30kDataset(Sequence):
    """
    Constructor for the dataset 
    Inputs 
    image_dir: (str) Directory for the images for the Dataset
    captions_file (str) Filename of the file with captions
    batch_size: (Optional[int]): Batch size used for batching of the dataset. 32 is the default value 
    preprocessing: Arr[str]: Types of preprocessing of the images that should occur 
    """
    def __init__(self, image_dir, captions_file, batch_size=32, preprocessing=['resize', 'normalize', 'augment'], tokenize=True,feature_extractor="InceptionV3", split_ratio=0.8, pickle_file=True):
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
        
        self.model = None
        
        self.split_ratio = split_ratio
        
        self.train_image_ids, self.test_image_ids = self.split_data()
        
        # Adding a feature extractor attribute
        self.feature_extractor = feature_extractor

        pickle_path = f"flickr8k_{str(self.feature_extractor)}"
        # Loading pre-extracted image features or extracting them
        if pickle_file and os.path.exists(pickle_path):
            print("Loading pickle file")
            self.load_pickle_data(pickle_path)
        else:
            self.images = self.pre_extract_image_features(self.images)
            if not pickle_file:
                self.save_pickle_data(pickle_path)
        
    
    def preprocess_input(self, image):
        if self.feature_extractor == "VGG16":
            image = vgg_preprocess(image)
        elif self.feature_extractor == "ResNet50":
            image = resnet_preprocess(image)
        elif self.feature_extractor == "InceptionV3":
            image = inception_preprocess(image)
        
        
        
        return image

    def extract_image_features(self, image):
        img = np.expand_dims(image, axis=0)
        img = self.preprocess_input(img)
        features = self.model.predict(img)
        features = GlobalAveragePooling2D()(features)
        return features

    def pre_extract_image_features(self, images):
        print("Extracting features d")
        if self.feature_extractor == "VGG16":
            base_model = VGG16(weights='imagenet', include_top=False)
        elif self.feature_extractor == "ResNet50":
            base_model = ResNet50(weights='imagenet', include_top=False)
        elif self.feature_extractor == "InceptionV3":
            base_model = InceptionV3(weights='imagenet', include_top=False)

        self.model = Model(inputs=base_model.input, outputs=base_model.output)
        
        with ThreadPoolExecutor() as executor:
            image_features = list(executor.map(self.extract_image_features, images))

        return image_features
    
    def split_data(self):
        total_images = len(self.image_ids)
        split_index = int(total_images * self.split_ratio)
        random.shuffle(self.image_ids)
        train_image_ids = self.image_ids[:split_index]
        test_image_ids = self.image_ids[split_index:]
        return train_image_ids, test_image_ids
    
    def get_all_data(self):
        print("get all data")
        all_image_ids = self.train_image_ids + self.test_image_ids
        all_indices = [self.index_dict[image_id] for image_id in all_image_ids]
        all_image_features = [self.images[i] for i in all_indices]
        all_unprocessed_captions = [self.captions[image_id] for image_id in all_image_ids]
        label_unprocessed_captions = [random.choice(caption_list) for caption_list in all_unprocessed_captions]

        all_captions = [self.process_caption(caption) for caption in all_unprocessed_captions]
        all_captions = [random.choice(caption_list) for caption_list in all_captions]

        return np.array(all_image_features), np.array(all_unprocessed_captions), np.array(label_unprocessed_captions)
        
    def get_all_captions(self):
        print("Get all captions")
        all_image_ids = self.train_image_ids + self.test_image_ids
        all_unprocessed_captions = [self.captions[image_id] for image_id in all_image_ids]
        all_captions_flat = [caption for caption_list in all_unprocessed_captions for caption in caption_list]
        return all_captions_flat
    def get_all_train_data(self):
        print("get all train data")
        train_indices = [self.index_dict[image_id] for image_id in self.train_image_ids]
        train_image_features = [self.images[i] for i in train_indices]
        train_unprocessed_captions = [self.captions[image_id] for image_id in self.train_image_ids]
        train_unprocessed_captions = [random.choice(caption_list) for caption_list in train_unprocessed_captions]
        
        train_captions = [self.process_caption(caption) for caption in train_unprocessed_captions]
        train_captions = [random.choice(caption_list) for caption_list in train_captions]

        return np.array(train_image_features), np.array(train_unprocessed_captions), np.array(train_captions)
    
    def get_all_test_data(self):
        print("get all test data")
        test_indices = [self.index_dict[image_id] for image_id in self.test_image_ids]
        test_image_features = [self.images[i] for i in test_indices]
        test_unprocessed_captions = [self.captions[image_id] for image_id in self.test_image_ids]
        test_unprocessed_captions = [random.choice(caption_list) for caption_list in test_unprocessed_captions]
        test_captions = [self.process_caption(caption) for caption in test_unprocessed_captions]
        test_captions = [random.choice(caption_list) for caption_list in test_captions]

        return np.array(test_image_features), np.array(test_unprocessed_captions), np.array(test_captions)
    
    """
    Pickle the data
    """
    def save_pickle_data(self, pickle_file):
        with open(pickle_file, 'wb') as f:
            pickle.dump((self.images, self.captions, self.image_ids, self.index_dict, self.mean, self.std), f)

    def load_pickle_data(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            self.images, self.captions, self.image_ids, self.index_dict, self.mean, self.std = pickle.load(f)
        
       
    
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
    Purpose: This function processes the images with various image processing techniques
    Input: Image --> image to edit 
    Output: Image --> processed image
    """
    def process_image(self, image):
        return image
    
    """
    Purpose: To extract the captions from the caption file and return them as an index dictionary with photo_id and captions
    Input: caption_file --> str --> representing the file with the captions
    Output: captions --> dict[str] --> Dictionary mapping photos to 5 captions
    """
    def load_captions(self, caption_file):
        captions_data = pd.read_csv(caption_file, delimiter='|')
        image_captions = {}

        def process_row(index, row):
            image_id = row.image_name.strip()
            caption = str(row.comment).strip()
            return image_id, caption

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_row, captions_data.index, captions_data.itertuples(index=False)))

        for index, (image_id, caption) in enumerate(results):
            if index % 20000 == 0:
                print(f"Processing caption {index}/{len(captions_data)}")
            if image_id not in image_captions:
                image_captions[image_id] = []
            image_captions[image_id].append(caption)

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
        batch_image_features = [self.images[i] for i in batch_indices]
        batch_unprocessed_captions = [self.captions[image_id] for image_id in batch_image_ids]

        batch_captions = [self.process_caption(caption) for caption in batch_unprocessed_captions]

        return np.array(batch_image_features), np.array(batch_unprocessed_captions), np.array(batch_captions)