import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os 
import numpy as np
from transformers import BertTokenizer

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
"""
This class handles the Processing of Text to build a vocabulary, using Keras Text Tokenizer and Padding Sequences
The only tokens we use are start, end, OOV and pad tokens.

#TODO: 
Currently, spacy preprocessing is in coco120k, flickr8k and flickr30k. But it is probably better here. When everything
is working, we can change it and get it working. I'm not sure exactly what's the best standard
"""
class BERTTextPreprocessor:
    """
    Constructor for Processor
    All_captions: List of captions --> up to 5 captions
    Max_caption_length: Max token length of a caption
    Vocab_size: Size of the vocabulary used --> This will come into help when using embeddings 
    use_spacy: Whether to use spacy
    """
    def __init__(self, all_captions, max_caption_length=15, vocab_size=20000, use_spacy=True, pickle='tokenizer.pkl', pickle_action=False):
        self.use_spacy = use_spacy
        self.captions = self.process_captions(all_captions)
        
        # Load the pre-trained BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_tokens(['<start>', '<end>', '<pad>', '<OOV>'])
        self.tokenizer.padding_side = "left"

        self.vocab_size = vocab_size
        self.max_caption_length = max_caption_length

        # Adding 4 because we have four special tokens - <pad>, <start>, <end>, <OOV>
        # self.tokenizer.vocab_size += 4
        # self.tokenizer.get_vocab()['<start>'] = self.tokenizer.vocab_size - 4
        # self.tokenizer.get_vocab()['<end>'] = self.tokenizer.vocab_size - 3
        # self.tokenizer.get_vocab()['<pad>'] = self.tokenizer.vocab_size - 2
        # self.tokenizer.get_vocab()['<OOV>'] = self.tokenizer.vocab_size - 1

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    # getter method for captions
    def get_captions(self):
        return self.captions

    """
    Input: all captions --> list of captions
    Output: Processed/tokenized captions based on format
    """
    def process_captions(self, all_captions):
        print("Processing caption")
        processed_captions = []
        for idx, image_captions in enumerate(all_captions):
            if idx % 1000 == 0:
                print(f"Processing {idx} out of {len(all_captions)}")
            for cap in image_captions:
                if self.use_spacy:
                    tokens = self.spacy_preprocess(cap)
                elif isinstance(cap, list):
                    tokens = cap
                else:
                    tokens = cap.split()
                processed_captions.append(' '.join(tokens))
        return processed_captions

    def spacy_preprocess(self, text):
        doc = nlp(text.lower())
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and not token.like_num
        ]
        return tokens
    """
    Tokenize and pad the captions
    """
    def tokenize(self, captions):
        captions = ['<start>' + str(caption) + ' <end>' for caption in captions]
        sequences = [self.tokenizer.encode(caption, add_special_tokens=True, max_length=self.max_caption_length, truncation=True) for caption in captions]
        return sequences
    """
    detokenizes sequence. This will help to visualize our model output at the end
    """
    def detokenize(self, sequences):
        def get_word(t):
            if isinstance(t, (list, np.ndarray)):  # If t is a list or an array
                return ' '.join(self.tokenizer.decode(i, skip_special_tokens=True) for i in t if i != 0)
            else:  # If t is a single value
                return self.tokenizer.decode(t, skip_special_tokens=True) if t != 0 else ''

        caption = ' '.join(get_word(token) for token in sequences)
        return caption
        

    # Save the tokenizer to a file
    def save_tokenizer(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

    # Load the tokenizer from a file
    @staticmethod
    def load_tokenizer(file_path):
        with open(file_path, 'rb') as f:
            loaded_tokenizer = pickle.load(f)
        return loaded_tokenizer