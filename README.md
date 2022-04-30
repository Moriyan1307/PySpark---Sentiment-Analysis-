# PySpark---Sentiment-Analysis-

# Twitter Sentiment Analysis using PySpark
## Getting Started with Jupyter Notebook

Accessed the Twitter API for live streaming tweets. Performed Feature Extraction and transformation from the JSON format of tweets using machine learning package of python pyspark.mllib. Experimented with three classifiers -Naïve Bayes, Logistic Regression and Decision Tree Learning and performed k-fold cross validation to determine the best.

You Have to Install Jupyter on your Local machine with:
>pip install jupyterlab

Once installed, launch JupyterLab with:
>jupyter-lab

## Jupyter Notebook
Install the classic Jupyter Notebook with:
>pip install notebook

To run the notebook:
>jupyter notebook

## Cleaning:
Firstly , the csv file is read with a csv file reader. The polarity of the tweets and the tweet text itself are extracted from the csv and stored in two seperate lists. We extract each word from the tweet text in order to do further cleaning. This is achieved by an easy spilt with a space i.e, split(" "). The words obtained by splitting is stored in another word list for easy access.


## Important Note : We will work in local mode with my laptop. The local mode is often used for prototyping, development, debugging, and testing. However, as Spark's local mode is fully compatible with the cluster mode, codes written locally can be run on a cluster with just a few additional steps.

## Computing Term Frequency :
According to the documentation : "TF and IDF are implemented in HashingTF and IDF. HashingTF takes an RDD of list as the input. Each record could be an iterable of strings or other types." Through my previous attempt of sentiment analysis with Pandas and Scikit-Learn, I learned that TF-IDF with Logistic Regression is quite strong combination, and showed robust performance, as high as Word2Vec + Convolutional Neural Network model. So in this post, I will try to implement TF-IDF + Logistic Regression model with Pyspark.

## Computing Term Frequency :
According to the documentation : "TF and IDF are implemented in HashingTF and IDF. HashingTF takes an RDD of list as the input. Each record could be an iterable of strings or other types."

## CountVectorizer + IDF + Logistic Regression
There's another way that you can get term frequecy for IDF (Inverse Document Freqeuncy) calculation. It is CountVectorizer in SparkML. Apart from the reversibility of the features (vocabularies), there is an important difference in how each of them filters top features. In case of HashingTF it is dimensionality reduction with possible collisions. CountVectorizer discards infrequent tokens.

## TRAINING :
We have used 2 models for training as explained below :

## Naive Bayes :

Naive Bayes is a simple multiclass classification algorithm with the assumption of independence between every pair of features. Naive Bayes can be trained very efficiently. Within a single pass to the training data, it computes the conditional probability distribution of each feature given label, and then it applies Bayes’ theorem to compute the conditional probability distribution of label given an observation and use it for prediction.

## Logistic Regression :

Logistic Regression is widely used to predict a binary response.

## Using Pyspark.ml we evalute Accuracy(With chi squared)
roc_auc = evaluator.evaluate(predictions)

print "Accuracy Score: {0:.4f}".format(accuracy)
print "ROC-AUC: {0:.4f}".format(roc_auc)

## And we do it without chisquared
trigramwocs_pipelineFit = build_ngrams_wocs().fit(train_set)
predictions_wocs = trigramwocs_pipelineFit.transform(val_set)
accuracy_wocs = predictions_wocs.filter(predictions_wocs.label == predictions_wocs.prediction).count() / float(val_set.count())
roc_auc_wocs = evaluator.evaluate(predictions_wocs)

Using above function we calculate the Accuracy of the prediction

Try that model on the final dataset.

Findings :
Here , we understand that Logistic Regression has the highest Accuracy as well as Precision and Recall . By this we can infer that Logistic regression is more robust and efficeint in classifying binary labels. According to the graphs, it appears that the Naïve Bayes Classifier overfits the most. Essentially, we have come to this conclusion because the Training accuracy of Naïve Bayes is 81.29125 % and the Test accuracy of Naïve Bayes is 79.8232590529248 % .The difference is large which indicates that the classifier has classified the training data better than the test data, and therefore we can say that Naïve Bayes over-fits.]










