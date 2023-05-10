import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TimeDistributed
from keras.layers import Reshape
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras import layers

from nltk.translate.meteor_score import meteor_score
import matplotlib.pyplot as plt


class ImageCaptioningModelNoTransformLSTM:
    def __init__(self, processor, embedding_matrix, image_features, captions, search="Greedy"):
        print("Init model")
        self.processor = processor
        self.tokenizer = self.processor.tokenizer 
        self.embedding_matrix = embedding_matrix
        self.image_features = image_features
        self.image_features = np.reshape(self.image_features, (-1, 2048))
        self.captions = captions
        self.max_length = max(len(c) for c in captions)
        self.vocab_size = self.processor.get_vocab_size()
        self.embedding_dim = embedding_matrix.shape[1]
        self.lstm_layer_dims = [self.vocab_size, 1024, 528, 256]
        self.search = search
        self.model = self.build_model()
        self.num_transformer_blocks = 3

    def build_model(self):
        # Define the image feature input
        image_input = Input(shape=(2048,))

        # Define the caption input
        caption_input = Input(shape=(self.max_length-1,), dtype="int32")

        # Define the embedding layer
        embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, 
                                           weights=[self.embedding_matrix], trainable=False, mask_zero=True)
        
        # Embed the caption input
        embedded_caption = embedding_layer(caption_input)

        # Project the image features into the same dimension as the embedded captions
        image_features_dense = Dense(self.embedding_dim, activation="relu")(image_input)

        # Repeat the image features max_length - 1 times along a new axis
        repeated_image_features = RepeatVector(self.max_length - 1)(image_features_dense)

        # Connect the repeated image features and the embedded caption
        image_caption_input = layers.concatenate([repeated_image_features, embedded_caption], axis=2)
        
        
        
        # Define the LSTM layers
        lstm_layers = []
        for i in range(len(self.lstm_layer_dims)):
            lstm_layers.append(LSTM(self.lstm_layer_dims[i], return_sequences=True))
            lstm_layers.append(Dropout(0.5))
        
        # Apply the LSTM layers to the image-caption input
        lstm_output = image_caption_input
        for lstm_layer in lstm_layers:
            lstm_output = lstm_layer(lstm_output)

        # Define the output layer
        output_layer = TimeDistributed(Dense(self.vocab_size, activation="softmax"))

        # Connect the output layer to the LSTM output
        output = output_layer(lstm_output)

        # Define the model
        model = Model(inputs=[image_input, caption_input], outputs=output)

        # Define the loss function and optimizer
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()

        # Compile the model
        model.compile(loss=loss_function, optimizer=optimizer)

        return model
    

    def save_model(self, file_path='my_model_weights.h5'):
        self.model.save_weights(file_path)

    def load_model(self, file_path='my_model_weights.h5'):
        self.model.load_weights(file_path)
        print("Model weights loaded from ", file_path)
    
    def split_data_for_evaluate(self, test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(self.image_features, self.captions, test_size=test_size, random_state=42)
        # Prepare the input and output data for the model
        x_train_caption_input = y_train[:, :-1]
        x_test_caption_input = y_test[:, :-1]
        y_train_output = y_train[:, 1:]
        y_test_output = y_test[:, 1:]
        
        return x_train, x_test, x_train_caption_input, x_test_caption_input, y_train_output, y_test_output
        
    def train(self, epochs=10, batch_size=128, test_size=0.05):
        # Split the data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(self.image_features, self.captions, test_size=test_size, random_state=42)

        # Prepare the input and output data for the model
        x_train_caption_input = y_train[:, :-1]
        x_test_caption_input = y_test[:, :-1]
        y_train_output = y_train[:, 1:]
        y_test_output = y_test[:, 1:]
        

        num_batches = len(x_train) // batch_size
        print(f"Training on {num_batches} batches...")
        train_losses = []
        for epoch in range(3):
            print("Epoch", epoch)
            batch_loss = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                batch_x_train = x_train[start_idx:end_idx]
                batch_x_train_caption_input = x_train_caption_input[start_idx:end_idx]
                
                batch_y_train_output = y_train_output[start_idx:end_idx]
                
                if batch_x_train_caption_input.shape[1] == 1:
                    batch_x_train_caption_input = np.repeat(batch_x_train_caption_input, 14, axis=1)

                loss = self.model.train_on_batch([batch_x_train, batch_x_train_caption_input], batch_y_train_output)

                print(f"Processing batch {batch_idx}/{num_batches}")
                
                batch_loss.append(loss)
                train_losses.append(loss)
            
            train_losses.append(np.mean(np.array(batch_loss)))
        
        # # Plot the train loss
        # plt.plot(train_losses)
        # plt.title("Train Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.savefig("loss.png")

        self.save_model()
        
        
        self.evaluate(x_test, x_test_caption_input, y_test_output)


    def evaluate(self, x_test, x_test_caption_input, y_test_output, search="Greedy", load_model=False):
        
        if load_model:
            self.load_model()
        references = []
        predictions = []

        for idx, (image, caption_input) in enumerate(zip(x_test, x_test_caption_input)):
            true_caption = self.processor.detokenize(y_test_output[idx])
            if self.search == "Greedy":
                generated_caption = self.predict_greedy(image, caption_input)
            else:
                generated_caption = self.predict_beam_search(image, caption_input)
            
            print("array", true_caption)
            print("generated", generated_caption)
            
            references.append(true_caption)
            predictions.append(generated_caption)

            # # Prepare the references and predictions for BLEU calculation
            # references.append([self.processor.tokenizer.sequences_to_texts([true_caption])[0].split()[1:-1]])
            # predictions.append(generated_caption.split()[1:-1])

        # Compute the BLEU score
        bleu_score = corpus_bleu(references, predictions)
        print(f"BLEU score: {bleu_score}")

    def repeat_last_token(self, input_caption, max_length):
        # Calculate the number of times the last token needs to be repeated
        repeat_count = max_length - input_caption.shape[1]

        # Get the last token of the input_caption
        last_token = input_caption[:, -1]

        # Repeat the last token
        repeated_last_token = np.repeat(last_token, repeat_count)

        # Reshape the repeated_last_token to match the input_caption's shape
        repeated_last_token = repeated_last_token.reshape((input_caption.shape[0], repeat_count))

        # Concatenate the input_caption and repeated_last_token along the sequence axis (axis=1)
        repeated_input_caption = np.concatenate([input_caption, repeated_last_token], axis=1)

        return repeated_input_caption
    
    def predict_greedy(self, image, caption_input):
        # Initialize the generated caption with the <start> token
        generated_caption = [self.processor.tokenizer.word_index['<start>']]
        pad_token = self.processor.tokenizer.word_index['<pad>']
        min_caption_length = 7

        for i in range(self.max_length - 1):
            # Prepare the input for the model
            input_image = np.expand_dims(image, axis=0)
            input_caption = generated_caption + [pad_token] * (self.max_length - 1 - len(generated_caption))
            input_caption = np.expand_dims(generated_caption, axis=0)

            # # Repeat the last token of the input_caption
            # input_caption = self.repeat_last_token(input_caption, self.max_length - 1)

            # Generate the probabilities for the next token
            predictions = self.model.predict([input_image, input_caption])

            # Select the token with the highest probability
            predicted_token = np.argmax(predictions[0, i])

            # Filter out unwanted tokens and prevent consecutive repetitions
            while predicted_token == self.processor.tokenizer.word_index['<start>'] or predicted_token >= self.processor.vocab_size or (predicted_token == self.processor.tokenizer.word_index['<end>'] and len(generated_caption) < min_caption_length) or (predicted_token == generated_caption[-1]):
                predictions[0, i, predicted_token] = 0
                predicted_token = np.argmax(predictions[0, i])

            # Stop adding tokens to the caption once <end> token is predicted or max length is reached
            if predicted_token == self.processor.tokenizer.word_index['<end>'] or len(generated_caption) == self.max_length - 1:
                generated_caption.append(self.processor.tokenizer.word_index['<end>'])
                break

            # Add the predicted token to the generated caption
            generated_caption.append(predicted_token)

        if len(generated_caption) > self.max_length:
            generated_caption = generated_caption[:self.max_length-1] + self.processor.tokenizer.word_index['<end>']
        # Convert the generated caption to text
        caption_text = self.processor.detokenize(generated_caption)

        return caption_text

    
    def predict_beam_search(self, image, caption_input, beam_width=3, min_length=10):
        start_token = self.processor.tokenizer.word_index['<start>']
        end_token = self.processor.tokenizer.word_index['<end>']

        # Initialize the beams with the <start> token and their probabilities as 1
        initial_beam = [([start_token], 1.0)]
        beams = [initial_beam]

        # Perform beam search
        for _ in range(self.max_length - 1):
            new_beams = []
            for beam in beams:
                input_caption, beam_prob = beam
                input_caption_np = np.array(input_caption)

                # Prepare the input for the model
                input_image = np.expand_dims(image, axis=0)
                input_caption_exp = np.expand_dims(input_caption_np, axis=0)
                input_caption_exp = self.repeat_last_token(input_caption_exp, self.max_length - 1)

                # Generate the probabilities for the next token
                predictions = self.model.predict([input_image, input_caption_exp])

                # Extract the probabilities of the last token in the beam
                token_probs = predictions[0, len(input_caption) - 1]

                # Get the indices of the beam_width highest probabilities
                top_indices = token_probs.argsort()[-beam_width:][::-1]

                for index in top_indices:
                    # Skip if the index is the start token or if the token is repeated
                    if index == start_token or (input_caption and index == input_caption[-1]):
                        continue

                    # Skip if the index is the end token and the sequence is not long enough
                    if index == end_token and len(input_caption) < min_length:
                        continue

                    # Calculate the new beam probability
                    new_prob = beam_prob * token_probs[index]

                    # Create a new beam with the predicted token and its probability
                    new_beam = (input_caption + [index], new_prob)
                    new_beams.append(new_beam)

            # Keep only the top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        # Select the beam with the highest probability
        best_beam = beams[0]

        # Convert the best beam to text
        caption_text = self.processor.detokenize(best_beam[0])

        return caption_text