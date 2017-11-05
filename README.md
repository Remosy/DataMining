# DataMining :page_facing_up:

- [x] Proposal 5pm 18th Aug 2017
- [x] Presentation or Demo in week 12
- [x] Report 5pm 20th Oct 2017

# Proposal :white_check_mark:

## Application
There are two sides of this project we are interested in. One side, as online peer-to-peer electronic business, both positive and negative of customer comments are valuable to analysis, and they could be the indicator of developing service quality, customer satisfaction and loyalty. On the other side, thousands and millions of pieces of comments are required time and money. To reduce such cost of analysis, technologies are indispensable. 

As we know, data mining is data driven solution and widely used in many industries.To recognise some patterns behind big data, there are massive algorithms of data mining that support auto-analysis. Among of these, the framework of data mining helps organise and break down our problem. The positivity or negativity of a customer comment will be a binary-class of this project, data mining regards it as a classification problem. Once we triggered the concept of data mining, next is choosing the text mining as the data mining application. 

In this project, we will work on text mining by Python and use the approaches that provided by Kotzias’s paper, which used the same database as us. According to his paper, we will use deep learning algorithms[1]. There are also several software applications will help with. Tensorflow is an open-source deep learning library, it has full of CNN(Convolutional Neural Network) algorithms for Python. Keras 2.0 will be the Python API, used for Tensorflow. It’s user friendly, modulable, and extensible. Since the customer comments are semantic, and according to Kotzias’s methods[1], Gensim will help transform the texts of a sentence into vector matrix, and it is adaptable to input of word2vec algorithm, provided by Tensorflow. 

## Datasets		

For this project, we will be using a dataset containing sentences labelled with positive and negative sentiment. A sentiment score is associated with each sentence and based on that the sentence is classified as positive and negative. Score is either 1 for positive and 0 for negative. These sentences are actual user reviews that are extracted from the different websites like amazon, imdb and yelp. Moreover, the sentences are very useful as they will be used to test and train our classifier. 

After searching, we got the data set from UCI machine learning repository although we searched few other websites but the data from UCI website was quite genuine and relevant to our project. This data set contains 500 positive and 500 negative user reviews from 3 different website. In total, we have 3000 sentences already classified into positive and negative sentences. We don’t have any numeric variable except the score. The attribute is text sentences. 

In the sentiment analysis data set we will review the user comments from the top websites like amazon, yelp and imdb. We will give scores on an integer scale from 1 to 5 and we consider scores of 4 and 5 to be positive and scores of 1 and 2 to be negative. We randomly divide the data into half, one for training and other for testing the documents in each dataset. Each review or comment is labelled and has a binary sentiment label either positive or negative then we can analyse the data easily. We will divide only the labelled data into half that is training and testing and in this project we train only on the labelled part of training set and we label the unlabelled data manually in each dataset by 50% positive and 50% negative. 

## Algorithm and Implementations		
The task our team will be performing is Classification. At the end, our main goal is to check the sentiment associated with the sentences and classify them into positive and negative. For now, we are not considering the neutral sentences.

The problem is in the scope of text mining. 

The project will be implemented in python.

We will build a vocabulary of our dataset using Gensim. Then to vectorize every word with word2vec algorithm which Gensim provided. Sentences will be represented as concatenated word vectors. A 2D convolutional neural network will be trained to classify sentences to positive or negative[2]. The neural network will be implemented using keras with tensorflow background.

## Evaluations	
In our case, evaluation strategies are measurements of performance of our classifier. Naively we will use accuracy as our main metric since the distribution of classes in our data is uniform. To measure the success of CNN model. We will also construct a 2x2 confusion matrix to check whether our classifier performs consistently for both classes.

|                      | Predicted Positive | Predicted Negative |
| -------------------- | ------------------ |------------------|
| Actual Positive      | TN                 | FP               |
| Actual Negative      | FN                 | TP               |


True positives (TP): These are comments in which we predicted negative, and actually they are negative as well.

True negatives (TN): We predicted positive, and actually, they are positive as well.

False positives (FP): We predicted negative, but actually, they are positive.

False negatives (FN): We predicted positive, but they actually are negative. 


However, our classifier is only about two classes, and will be trained via CNN, thus ROC (Receiver Operating Characteristics) may not be used to evaluate our results. 
In the end, we will use some new data to test our model. higher accuracy means better performance of our classification model.

## References
[1] Kotzias et. al, '*From Group to Individual Labels using Deep Features*', KDD, 2015 

[2] Kim, ‘*Convolutional Neural Networks for Sentence Classification*’, arXiv, 2014

## Appendix	:link:						
Deep learning software library: TensorFlow 1.2

Website:https://www.tensorflow.org

Python Deep learning library: Keras 2.0

Website: https://keras.io

Fast Vector Space Modelling Python library: Gensim

Website: https://pypi.python.org/pypi/gensim

Data Sets:

Website: http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences?fref=gc

