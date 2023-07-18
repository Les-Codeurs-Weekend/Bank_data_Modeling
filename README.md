# Bank_data_Modeling

This Jupyter notebook performs a bank marketing analysis using various classifiers and a neural network. The dataset used in this analysis is stored in the "bank-full.csv" file, and it contains information about bank customers and whether they subscribed to a term deposit (binary classification problem).
# Libraries Used

The analysis starts by importing the necessary libraries for data processing, model training, and evaluation. Some of the key libraries used include:

    NumPy: For numerical operations.
    Pandas: For data manipulation and analysis.
    Scikit-learn: For building machine learning models and evaluating their performance.
    Keras: For building the neural network.

# Data Preprocessing

The dataset is loaded from "bank-full.csv" into a Pandas DataFrame. Some data preprocessing steps are applied to convert categorical variables into numerical format using label encoding. The target variable 'y' (whether the customer subscribed to a term deposit) is converted to binary labels (0 or 1).
Exploratory Data Analysis

Basic data exploration is performed to understand the distribution of the target variable and the feature variables. Various classifiers are then trained using the preprocessed data.
Classifiers

    Dummy Classifier: Two dummy classifiers are used, one with "most_frequent" strategy and another with "uniform" strategy. These classifiers serve as baseline models for comparison.

    K-Nearest Neighbors (KNN) Classifier: A KNN classifier with k=10 is trained.

    Gaussian Naive Bayes Classifier: A Gaussian Naive Bayes classifier is trained.

    Random Forest Classifier: A Random Forest classifier is trained.

# Neural Network

A simple feedforward neural network with four hidden layers is built using Keras. The input layer has 16 neurons (matching the number of features), and each hidden layer has 16 neurons with batch normalization and ELU activation. The output layer has two neurons (binary classification) with sigmoid activation.

The neural network is trained using the Adam optimizer and categorical cross-entropy loss function.
Evaluation

Each classifier's performance is evaluated using accuracy, precision, recall, and F1-score on the test set. For the neural network, one-hot encoding is applied to the target variable for training, and argmax is used to convert the predicted probabilities back to binary labels for evaluation.

# Conclusion

The analysis demonstrates the performance of various classifiers and a neural network on the bank marketing dataset. Depending on the specific evaluation metric and use case, different classifiers may be more suitable for predicting term deposit subscriptions.

For more details and specific code snippets, please refer to the Jupyter notebook.
