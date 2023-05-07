import tensorflow as tf

class CNNEncoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNNEncoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x