import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
"""
This class handles the Processing of Text to build a vocabulary, using Keras Text Tokenizer and Padding Sequences
The only tokens we use are start, end, OOV and pad tokens.

#TODO: 
Currently, spacy preprocessing is in coco120k, flickr8k and flickr30k. But it is probably better here. When everything
is working, we can change it and get it working. 
"""
class TextPreprocessor:
    """
    Constructor for Processor
    All_captions: List of captions --> up to 5 captions
    Max_caption_length: Max token length of a caption
    Vocab_size: Size of the vocabulary used --> This will come into help when using embeddings 
    use_spacy: Whether to use spacy
    """
    def __init__(self, all_captions, max_caption_length=15, vocab_size=5000, use_spacy=False):
        self.use_spacy = use_spacy
        self.captions = self.process_captions(all_captions)
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<unk>', filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        self.tokenizer.fit_on_texts(self.captions)
        self.vocab_size = vocab_size
        self.max_caption_length = max_caption_length

        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'
    
    # getter method for captions
    def get_captions(self):
        return self.captions
    """
    Input: all captions --> list of captions
    Output: Processed/tokenized captions based on format
    """
    def process_captions(self, all_captions):
        processed_captions = []
        for image_captions in all_captions:
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
        captions = ['<start> ' + caption + ' <end>' for caption in captions]
        sequences = self.tokenizer.texts_to_sequences(captions)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_caption_length, padding='post')
        return padded_sequences
    """
    detokenizes sequence. This will help to visualize our model output at the end
    """
    def detokenize(self, sequences):
        captions = []
        for sequence in sequences:
            caption = ' '.join(self.tokenizer.index_word[token] for token in sequence if token != 0)
            captions.append(caption)
        return captions