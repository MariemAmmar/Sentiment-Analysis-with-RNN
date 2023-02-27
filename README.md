# Sentiment Analysis with RNN
This code performs sentiment analysis on the IMDB movie reviews dataset using LSTM. The dataset contains 50,000 movie reviews, and each review is labeled as positive or negative. The code preprocesses the data by removing HTML tags, punctuation, and numbers, and then tokenizes the sentences. The GloVe embeddings are used to create the feature matrix, and the LSTM model is trained on the preprocessed data.

## Dependencies

* Pandas
* Numpy
* re
* NLTK
* Keras
* Tensorflow
* Seaborn
* Scikit-learn
* Matplotlib

## Dataset
The dataset used is the IMDB movie reviews dataset, which can be downloaded from [here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). It contains 50,000 movie reviews, each labeled as positive or negative.

## Preprocessing
* The preprocess_text() function removes HTML tags, punctuation, and numbers from the text. It also removes single-character words and multiple spaces.

* The remove_tags() function removes HTML tags from the text.

* The Tokenizer class from Keras is used to tokenize the sentences. The fit_on_texts() method is used to fit the tokenizer on the training data, and the texts_to_sequences() method is used to convert the sentences into sequences of integers.

* The pad_sequences() function is used to pad the sequences to a fixed length of 100.

## Embeddings
The glove.6B.100d.txt file  is used to create the feature matrix. The file contains pre-trained word embeddings, and the embeddings for each word in the vocabulary are extracted and used to create the feature matrix.
To use pre-trained GloVe embeddings, you can download the dataset from [here](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt).

## Model
The LSTM model consists of an embedding layer, an LSTM layer, and a dense layer. The embedding layer takes the feature matrix as input, and the LSTM layer processes the sequences. The dense layer produces the output, which is a probability between 0 and 1 that represents the sentiment of the input text.

## Training
The model is compiled using the adam optimizer and binary_crossentropy loss function. The fit() method is used to train the model on the preprocessed data.

## Evaluation
The model is evaluated on the test set using the evaluate() method.

## Prediction
The model is used to predict the sentiment of a single input text using the predict() method.

## Visualization
The training and validation accuracy and loss are plotted using the matplotlib library.

## Conclusion
This code performs sentiment analysis on the IMDB movie reviews dataset using LSTM. The model achieves a test accuracy of approximately 85%, indicating that it is effective at predicting the sentiment of movie reviews.
