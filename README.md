# Therapia-Thesis Project

## Introduction
This repository contains the implementation of machine learning models used in our thesis project to detect depression from text data. The aim of the project is to analyze user input to determine the likelihood of depression using Natural Language Processing (NLP) and machine learning techniques. The project includes dataset preprocessing, model training, and integration of the trained model into a web application for real-time analysis.

## Dataset
The dataset used in this project is in JSON format. It consists of text samples labeled as "depressed" or "not depressed". The JSON file should be placed in the root directory of the project. Each entry in the dataset has two fields: `text` and `class`, where `text` is the sample text, and `class` is the label (0 for not depressed, 1 for depressed).

## Preprocessing
The text data is preprocessed using the following steps:

- **Stop Words Removal:** Turkish stop words are removed from the text using NLTK's stop words corpus.
- **Vectorization:** The text data is converted into numerical format using `CountVectorizer` from scikit-learn. The vectorizer is configured to remove stop words and limit the number of features to 3500.

## Model Training
Two machine learning models were trained on the preprocessed data:

### Naive Bayes
The `MultinomialNB` model was used to train on the vectorized text data. The model's performance is evaluated using accuracy score and classification report.

### Support Vector Machine (SVM)
The `SVC` model with a regularization parameter `C=10` was trained on the same data. The model's performance is similarly evaluated.

## Real-Time Analysis
The trained SVM model is integrated into a web application to perform real-time depression analysis. Users can input sentences through the web interface, and the model predicts whether the input indicates depression.


## Contributors
- [@elifozyurek](https://github.com/elifozyurek)
- [@cerencaglayan](https://github.com/cerencaglayan)
- [@bilgehanay](https://github.com/bilgehanay)
- [@hakanTasar](https://github.com/hakanTasar)
