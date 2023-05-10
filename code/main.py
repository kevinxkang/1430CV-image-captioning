from flickr8k import Flickr8kDataset
# from flickr30k import Flickr30kDataset
# from coco120k import Coco120KDataset
from tokenizer import TextPreprocessor
from glove_embeddings import EmbeddingMatrix
from bert_tokenizer import BERTTextPreprocessor
import shutil
import os
import matplotlib.pyplot as plt
from ImageCaptioningModel import ImageCaptioningModel
from ImageCaptioningModelBERT import ImageCaptioningModelBERT
if __name__ == "__main__":
    flickr_8kdataset = Flickr8kDataset("/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/flickr8k_images/", "/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/flickr8k_images/captions.txt")
    # flickr_30kdataset = Flickr30kDataset("/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/flickr30k_images/", "/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/flickr30k_images/results.csv")
    # coco120k_imags = Coco120KDataset("/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/coco_120k/train2014/train2014/","/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/coco_120k/val2014/val2014/","/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/coco_120k/captions/annotations/captions_train2014.json", "/Users/onrmmm78/Desktop/cs1430/finalproject-mithunramesh19/data/coco_120k/captions/annotations/captions_val2014.json", max_samples=30000)
    
    image_features, all_captions, labels = flickr_8kdataset.get_all_data()
    image_features = image_features[:10]
    all_captions = all_captions[:10]
    labels = labels[:10]
    # train_data, test_data = flickr_8kdataset.get_all_train_data(), flickr_8kdataset.get_all_test_data()
    # plt.imshow(batch_images[0].astype("uint8"))
    # plt.show()
    # print("Captions for the first image:")
    # print("Captions for the first image:")
    # print(batch_unprocess_captions[0])
    # # print(batch_processed_captions[0])
    
    print("caption", all_captions)
    vocab_size = 10000
    # # Instantiate a TextPreprocessor object
    preprocessor = BERTTextPreprocessor(all_captions, vocab_size=vocab_size)
   
    # # Tokenize and pad the captions
    tokenized_captions = preprocessor.tokenize(labels)
    
    # # # Initialize the embedding matrix
    # embedding_matrix = EmbeddingMatrix(preprocessor, 'glove.6B.300d.txt')
    
    print("pre-tokenization", labels)
    print("post-tokenized", preprocessor.detokenize(tokenized_captions))
    print("decoded")
    
    # Create the ImageCaptioningModel
    model = ImageCaptioningModelBERT(preprocessor, None, image_features, tokenized_captions)
    
    # x_train, x_test, x_train_caption_input, x_test_caption_input, y_train_output, y_test_output = model.split_data_for_evaluate()
    # model.evaluate(x_test, x_test_caption_input, y_test_output, "Greedy", load_model=True)

    # Train and evaluate the model
    model.train(epochs=2, batch_size=1028)
    # untokenized_captions = preprocessor.detokenize(tokenized_captions)
    # # Print the tokenized captions
    # print(tokenized_captions[0])
    # print(untokenized_captions)
    
    # # # print(batch_unprocess_captions[0])
    # # print(batch_processed_captions[0])