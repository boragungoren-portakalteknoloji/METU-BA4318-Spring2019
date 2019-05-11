#!/usr/bin/env python
# coding: utf-8

# ### If you like this  kernel Greatly Appreciate if you can  UPVOTE .Thank you
# 
# # Introduction: Anomaly Detection 
# 
# Anomaly detection is a technique used to identify unusual patterns that do not conform to expected behavior, called outliers. It has many applications in business, from intrusion detection (identifying strange patterns in network traffic that could signal a hack) to system health monitoring (spotting a malignant tumor in an MRI scan), and from fraud detection in credit card transactions to fault detection in operating environments.
# 
# In this jupyter notebook we are going to take the credit card fraud detection as the case study for understanding this concept in detail.
# 
# ## What Are Anomalies?
# 
# Anomalies can be broadly categorized as:
# 
# ***Point anomalies:*** A single instance of data is anomalous if it's too far off from the rest. Business use case: Detecting credit card fraud based on "amount spent."
# 
# ***Contextual anomalies:*** The abnormality is context specific. This type of anomaly is common in time-series data. Business use case: Spending $100 on food every day during the holiday season is normal, but may be odd otherwise.
# 
# ***Collective anomalies:*** A set of data instances collectively helps in detecting anomalies. Business use case: Someone is trying to copy data form a remote machine to a local host unexpectedly, an anomaly that would be flagged as a potential cyber attack. 
# 
# - Anomaly detection is similar to — but not entirely the same as — noise removal and novelty detection. 
# 
# - ***Novelty detection*** is concerned with identifying an unobserved pattern in new observations not included in training data like a sudden interest in a new channel on YouTube during Christmas, for instance. 
# 
# - ***Noise removal (NR)*** is the process of removing noise from an otherwise meaningful signal. 
# 
# ## 1. Anomaly Detection Techniques
# #### Simple Statistical Methods
# 
# The simplest approach to identifying irregularities in data is to flag the data points that deviate from common statistical properties of a distribution, including mean, median, mode, and quantiles. Let's say the definition of an anomalous data point is one that deviates by a certain standard deviation from the mean. Traversing mean over time-series data isn't exactly trivial, as it's not static. You would need a rolling window to compute the average across the data points. Technically, this is called a ***rolling average or a moving average***, and it's intended to smooth short-term fluctuations and highlight long-term ones. Mathematically, an n-period simple moving average can also be defined as a ***"low pass filter."***
# 
# #### Challenges with Simple Statistical Methods
# 
# The low pass filter allows you to identify anomalies in simple use cases, but there are certain situations where this technique won't work. Here are a few:  
# 
# - The data contains noise which might be similar to abnormal behavior, because the boundary between normal and abnormal behavior is often not precise. 
# 
# - The definition of abnormal or normal may frequently change, as malicious adversaries constantly adapt themselves. Therefore, the threshold based on moving average may not always apply.
# 
# - The pattern is based on seasonality. This involves more sophisticated methods, such as decomposing the data into multiple trends in order to identify the change in seasonality.
# 
# ## 2. Machine Learning-Based Approaches
# 
# Below is a brief overview of popular machine learning-based techniques for anomaly detection. 
# 
# #### a.Density-Based Anomaly Detection 
# Density-based anomaly detection is based on the k-nearest neighbors algorithm.
# 
# Assumption: Normal data points occur around a dense neighborhood and abnormalities are far away. 
# 
# The nearest set of data points are evaluated using a score, which could be Eucledian distance or a similar measure dependent on the type of the data (categorical or numerical). They could be broadly classified into two algorithms:
# 
# ***K-nearest neighbor***: k-NN is a simple, non-parametric lazy learning technique used to classify data based on similarities in distance metrics such as Eucledian, Manhattan, Minkowski, or Hamming distance.
# 
# ***Relative density of data***: This is better known as local outlier factor (LOF). This concept is based on a distance metric called reachability distance.
# 
# #### b.Clustering-Based Anomaly Detection 
# Clustering is one of the most popular concepts in the domain of unsupervised learning.
# 
# Assumption: Data points that are similar tend to belong to similar groups or clusters, as determined by their distance from local centroids.
# 
# ***K-means*** is a widely used clustering algorithm. It creates 'k' similar clusters of data points. Data instances that fall outside of these groups could potentially be marked as anomalies.
# 
# #### c.Support Vector Machine-Based Anomaly Detection 
# 
# - A support vector machine is another effective technique for detecting anomalies. 
# - A SVM is typically associated with supervised learning, but there are extensions (OneClassCVM, for instance) that can be used to identify anomalies as an unsupervised problems (in which training data are not labeled). 
# - The algorithm learns a soft boundary in order to cluster the normal data instances using the training set, and then, using the testing instance, it tunes itself to identify the abnormalities that fall outside the learned region.
# - Depending on the use case, the output of an anomaly detector could be numeric scalar values for filtering on domain-specific thresholds or textual labels (such as binary/multi labels).
# 
# 
# In this jupyter notebook we are going to take the credit card fraud detection as the case study for understanding this concept in detail using the following Anomaly Detection Techniques namely
# 
# #### Isolation Forest Anomaly Detection Algorithm
# 
# #### Density-Based Anomaly Detection (Local Outlier Factor)Algorithm
# 
# #### Support Vector Machine Anomaly Detection Algorithm 
# 

# ## Credit Card Fraud Detection
# 
# ## Problem Statement:
# 
# The Credit Card Fraud Detection Problem includes modeling past credit card transactions with the knowledge of the ones that turned out to be fraud. This model is then used to identify whether a new transaction is fraudulent or not. Our aim here is to detect 100% of the fraudulent transactions while minimizing the incorrect fraud classifications.
# 
# #### DataSet : 
# 
# The dataset that is used for credit card fraud detection is derived from the following Kaggle URL :
# 
# https://www.kaggle.com/mlg-ulb/creditcardfraud
# 
# 

# #### Observations
# 
# - The data set is highly skewed, consisting of 492 frauds in a total of 284,807 observations. This resulted in only 0.172% fraud cases. This skewed set is justified by the low number of fraudulent transactions.
# 
# - The dataset consists of numerical values from the 28 ‘Principal Component Analysis (PCA)’ transformed features, namely V1 to V28. Furthermore, there is no metadata about the original features provided, so pre-analysis or feature study could not be done.
# 
# - The ‘Time’ and ‘Amount’ features are not transformed data.
# 
# - There is no missing value in the dataset.

# ## Preprocessing 
# 
# ### Import Libraries 

# In[ ]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot


# In[ ]:


data = pd.read_csv('../input/creditcard.csv',sep=',')

print(data.columns)


# In[ ]:


data1= data.sample(frac = 0.1,random_state=1)

data1.shape


# In[ ]:


data.describe()


# ## Exploratory Data Analysis

# In[ ]:


data.shape


# Let us now check the missing values in the dataset

# In[ ]:


data.isnull().values.any()


# In[ ]:


data.head()


# In[ ]:


count_classes = pd.value_counts(data['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency");


# Determine the number of fraud and valid transactions in the entire dataset.

# In[ ]:


Fraud = data[data['Class']==1]

Normal = data[data['Class']==0]


# In[ ]:


Fraud.shape


# In[ ]:


Normal.shape


# How different are the amount of money used in different transaction classes?

# In[ ]:


Fraud.Amount.describe()


# In[ ]:


Normal.Amount.describe()


# Let's have a more graphical representation of the data

# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(Fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(Normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# Do fraudulent transactions occur more often during certain time frame ? Let us find out with a visual representation.

# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(Fraud.Time, Fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(Normal.Time, Normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# In[ ]:


init_notebook_mode(connected=True)
plotly.offline.init_notebook_mode(connected=True)


# In[ ]:


# Create a trace

trace = go.Scatter(
    x = Fraud.Time,
    y = Fraud.Amount,
    mode = 'markers'
)
data = [trace]


# In[ ]:


plotly.offline.iplot({
    "data": data
})


# Doesn't seem like the time of transaction really matters here as per above observation.
# Now let us take a sample of the dataset for out modelling and prediction

# In[ ]:


data1.shape


# Plot histogram of each parameter

# In[ ]:


data1.hist(figsize=(20,20))
plt.show()


# Determine the number of fraud and valid transactions in the dataset.

# In[ ]:


Fraud = data1[data1['Class']==1]

Valid = data1[data1['Class']==0]

outlier_fraction = len(Fraud)/float(len(Valid))


# Now let us print the outlier fraction and no of Fraud and Valid Transaction cases 

# In[ ]:


print(outlier_fraction)

print("Fraud Cases : {}".format(len(Fraud)))

print("Valid Cases : {}".format(len(Valid)))


# Correlation Matrix

# In[ ]:


correlation_matrix = data1.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_matrix,vmax=0.8,square = True)

plt.show()


# The above correlation matrix shows that none of the V1 to V28 PCA components have any correlation to each other however if we observe Class has some form positive and negative correlations with the V components but has no correlation with Time and Amount.

# Get all the columns from the dataframe

# In[ ]:


columns = data1.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
target = "Class"
# Define a random state 
state = np.random.RandomState(42)
X = data1[columns]
Y = data1[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)


# ## Model Prediction

# Now it is time to start building the model .The types of algorithms we are going to use to try to do anomaly detection on this dataset are as follows
# 
# #### 1. Isolation Forest Algorithm: 
# One of the newest techniques to detect anomalies is called Isolation Forests. The algorithm is based on the fact that anomalies are data points that are few and different. As a result of these properties, anomalies are susceptible to a mechanism called isolation.
# 
# This method is highly useful and is fundamentally different from all existing methods. It introduces the use of isolation as a more effective and efficient means to detect anomalies than the commonly used basic distance and density measures. Moreover, this method is an algorithm with a low linear time complexity and a small memory requirement. It builds a good performing model with a small number of trees using small sub-samples of fixed size, regardless of the size of a data set.
# 
# Typical machine learning methods tend to work better when the patterns they try to learn are balanced, meaning the same amount of good and bad behaviors are present in the dataset.
# 
# #### How Isolation Forests Work
# 
# The Isolation Forest algorithm isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. The logic argument goes: isolating anomaly observations is easier because only a few conditions are needed to separate those cases from the normal observations. On the other hand, isolating normal observations require more conditions. Therefore, an anomaly score can be calculated as the number of conditions required to separate a given observation.
# 
# The way that the algorithm constructs the separation is by first creating isolation trees, or random decision trees. Then, the score is calculated as the path length to isolate the observation.
# 
# #### 2. Local Outlier Factor(LOF) Algorithm
# 
# The LOF algorithm is an unsupervised outlier detection method which computes the local density deviation of a given data point with respect to its neighbors. It considers as outlier samples that have a substantially lower density than their neighbors.
# 
# The number of neighbors considered, (parameter n_neighbors) is typically chosen 1) greater than the minimum number of objects a cluster has to contain, so that other objects can be local outliers relative to this cluster, and 2) smaller than the maximum number of close by objects that can potentially be local outliers. In practice, such informations are generally not available, and taking n_neighbors=20 appears to work well in general.
# 

# Define the outlier detection methods

# In[ ]:


classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction,random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1, random_state=state)
   
}


# Fit the model

# In[ ]:


n_outliers = len(Fraud)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:    
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != Y).sum()
    # Run Classification Metrics
    print("{}: {}".format(clf_name,n_errors))
    print("Accuracy Score :")
    print(accuracy_score(Y,y_pred))
    print("Classification Report :")
    print(classification_report(Y,y_pred))


# #### Observations :
# - Isolation Forest detected 73 errors versus Local Outlier Factor detecting 97 errors vs. SVM detecting 8516 errors
# - Isolation Forest has a 99.74% more accurate than LOF of 99.65% and SVM of 70.09
# - When comparing error precision & recall for 3 models , the Isolation Forest performed much better than the LOF as we can see that the detection of fraud cases is around 27 % versus LOF detection rate of just 2 % and SVM of 0%.
# - So overall Isolation Forest Method performed much better in determining the fraud cases which is around 30%.
# - We can also improve on this accuracy by increasing the sample size or use deep learning algorithms however at the cost of computational expense.We can also use complex anomaly detection models to get better accuracy in determining more fraudulent cases

# Now let us look at one particular Deep Learning Algorithm called ***Autoencoders***
# 

# ## Autoencoders
# 
# An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner.
# 
# The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for the purpose of dimensionality reduction.
# 
# An autoencoder learns to compress data from the input layer into a short code, and then uncompress that code into something that closely matches the original data. This forces the autoencoder to engage in dimensionality reduction, for example by learning how to ignore noise. Some architectures use stacked sparse autoencoder layers for image recognition. The first autoencoder might learn to encode easy features like corners, the second to analyze the first layer's output and then encode less local features like the tip of a nose, the third might encode a whole nose, etc., until the final autoencoder encodes the whole image into a code that matches (for example) the concept of "cat".An alternative use is as a generative model: for example, if a system is manually fed the codes it has learned for "cat" and "flying", it may attempt to generate an image of a flying cat, even if it has never seen a flying cat before.
# 
# The simplest form of an autoencoder is a feedforward, non-recurrent neural network very similar to the many single layer perceptrons which makes a multilayer perceptron (MLP) – having an input layer, an output layer and one or more hidden layers connecting them – but with the output layer having the same number of nodes as the input layer, and with the purpose of reconstructing its own inputs (instead of predicting the target value Y  given inputs X). Therefore, autoencoders are unsupervised learning models. 
# 
# 
# ## If you like this kernel greatly appreciate an UPVOTE 
# 
