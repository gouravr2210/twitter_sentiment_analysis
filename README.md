# twitter_sentiment_analysis
Sentiment Analysis with Twitter Data
This project aims to perform sentiment analysis on tweets using the NLTK (Natural Language Toolkit) library in Python. The sentiment analysis is conducted on a dataset of tweets collected from Twitter.

Overview
Sentiment analysis is the process of determining the sentiment expressed in a piece of text, whether it is positive, negative, or neutral. In this project, we analyze the sentiment of tweets using machine learning techniques.

Data Preprocessing: The tweets are preprocessed to remove noise and irrelevant information such as hyperlinks, hashtags, mentions, and special characters. Text normalization techniques are applied, including tokenization, stemming, and removing stopwords.

Feature Extraction: The TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique is employed to convert the preprocessed text into numerical feature vectors. This technique represents the importance of a word in a document relative to a collection of documents.

Model Training: A Multinomial Naive Bayes classifier is trained on the TF-IDF features to predict the sentiment of tweets. The classifier learns from the labeled dataset of positive, negative, and neutral tweets.

Evaluation: The performance of the trained model is evaluated using metrics such as accuracy score, F1-score, and confusion matrix. These metrics provide insights into the effectiveness of the sentiment analysis model.

TextBlob Integration: Additionally, sentiment analysis is performed using the TextBlob library to compare results with the machine learning approach. TextBlob provides a pre-trained sentiment analysis model based on a pattern analysis approach.
