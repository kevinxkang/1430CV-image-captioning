import numpy as np
import tensorflow as tf

class ImageCaptionModel(tf.keras.Model):

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def call(self, image_embeddings, captions):
        return self.decoder(image_embeddings, captions)

    def train(self):
        pass

    def test(self):
        pass


    
