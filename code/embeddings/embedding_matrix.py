import numpy as np
from tensorflow.keras.layers import Embedding
import pickle
import os

class EmbeddingMatrix:
    """
    This class generates the embedding matrix for a given tokenizer and embedding file
    """
    def __init__(self, tokenizer, embedding_file, embedding_dim=300, pickle='embedding_matrix.pkl', pickle_action=False):
        self.tokenizer = tokenizer
        self.embedding_file = embedding_file
        self.word_index = self.tokenizer.tokenizer.word_index
        
        if not os.path.exists(pickle) or pickle_action is False:
            self.embedding_matrix = self.generate_matrix(embedding_dim)
            self.save_embedding_matrix(pickle)
        elif pickle_action is True:
            self.embedding_matrix = self.load_embedding_matrix(pickle)
        
    def get_embedding(self, word):
        index = self.word_index.get(word)
        if index is None:
            return None
        else:
            return self.embedding_matrix[index]
    """
    This function generates the embedding matrix for the given word_index and embedding file
    """
    def generate_matrix(self, embedding_dim):
        print("generating embedding matrix")
        embeddings_index = {}
        with open(self.embedding_file, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((len(self.word_index) + 1, embedding_dim))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return embedding_matrix
    
    # Save the embedding matrix to a file
    def save_embedding_matrix(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.embedding_matrix, f)

    # Load the embedding matrix from a file
    @staticmethod
    def load_embedding_matrix(file_path):
        with open(file_path, 'rb') as f:
            loaded_embedding_matrix = pickle.load(f)
        return loaded_embedding_matrix