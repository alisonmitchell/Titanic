# Titanic: Machine Learning from Disaster  
The Titanic passenger manifest is a popular dataset for practice in solving supervised learning classification problems. The challenge is to build a predictive model to determine which passengers were more likely to survive. 

This notebook demonstrates Exploratory Data Analysis techniques and modelling concepts learned on a face-to-face machine learning course and from various resources including blogs, tutorials, documentation and textbooks.

Importing and performing descriptive statistics on the dataset using Pandas revealed missing data and categorical and numerical data types. Visualisation using Matplotlib and Seaborn showed the composition and distribution of data and the strength of correlation between features which would greatly influence chances of survival. 

The preprocessing stage comprised imputing missing values, feature encoding to transform categorical into numerical data, feature scaling using NumPy to apply a log transformation to remove impact of skewness, feature extraction and engineering of additional features, binning data into groups, and deletion of irrelevant features to reduce noise and avoid overfitting. The data was standardised using scikit-learn's StandardScaler() function to transform attributes to a standard Gaussian distribution with a mean of 0 and a standard deviation of 1.

During the modelling stage a performance metric was defined (Accuracy), and 14 classification models were evaluated using a 10-fold cross-validation method including decision-tree-based ensemble algorithms, k-nearest neighbors, Naive Bayes, support vector classifier, and a deep neural network model using Keras on a TensorFlow backend. Six models were selected for optimisation using the Grid Search technique and hyperparameters were tuned. Feature importance, accuracy score, confusion matrix, precision and recall score metrics were compared before selecting the Gradient Boosting classifier as the best performing model. 

## Data source
[Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data?test.csv) 

## Libraries 
Numpy, Pandas, Matplotlib, Seaborn, SciPy, Scikit-learn, Keras and TensorFlow.