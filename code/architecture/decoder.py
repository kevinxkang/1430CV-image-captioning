import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, AdditiveAttention, Concatenate
from tensorflow.keras.models import Model


class LSTMWithAttenion(tf.keras.Model):

    def __init__(self, embedding_dim, units, vocab_size, max_length):
        super(LSTMWithAttenion, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(units, return_sequences=True, return_state=True)
        self.dense = Dense(vocab_size)
        self.attenion = AdditiveAttention()
        self.concat = Concatenate(axis=-1)
        self.max_length = max_length


    def call(self, image_features, captions):

        caption_embeddings = self.embedding(captions)

        lstm_output, _, _ = self.lstm(caption_embeddings)

        context_vector, attention_weights  = self.attenion([lstm_output, image_features])

        output = self.concat([context_vector, lstm_output])

        logits = self.dense(output)

        return logits, attention_weights
    



        




