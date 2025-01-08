# Way Of Code Workshop: Training and Evaluating a Sentiment Classifier

## Scenario

You have been hired by a social media analytics company to build a sentiment classifier. Your task is to create a model that can accurately classify product reviews as either positive or negative sentiment, which will help the company analyze customer feedback.

## Task

Having worked out simple feature extraction for text and evaluation of classifiers, you will finally put it all together with a larger dataset and machine learning. You will need to implement

- Data loading (already implemented to save you time),
- Text preprocessing,
- Feature extraction,
- Training a classifier, and
- Evaluating predictions made by a trained classifier.

## Assignment scoring

Complete each function and method according to instructions. There is a test for each coding section. Make sure that your code passes all the tests. Passing the tests that you can see will mean you are on the right track.

## Specific steps

Implement evaluation functions which will be used to calculate performance metrics when comparing some predicted classes with the corresponding actual classes.

### Import packages

The required packages to import are listed in `requirements.txt`. You may install and import whatever packages help you solve the assignment. You are not expected to implement any particular solution from scratch.

**Hint:** `scikit-learn` is a popular Python library for machine learning. And `nltk` is a popular Python library for natural language processing.

### Data

The dataset (combined_sentiment_data.csv) contains product reviews with their associated sentiments. Each row contains:

```
sentence,sentiment
"Good case, Excellent value.",positive
"Tied to charger for conversations lasting more than 45 minutes.MAJOR PROBLEMS!!",negative
```

### Load data

The method `load_data` is already implemented to save you time. It returns four lists:

- train_data_raw: list of training data
- train_labels: list of training labels
- test_data_raw: list of test data
- test_labels: list of test labels

### Text preprocessing

The text data should be preprocessed in order to improve the quality of the feature extraction. Implement the function that does this for you. You have wide discretion in selecting which text preprocessing gives the best result when training and evaluating your classifier later.

You need to implement the function `preprocess` which takes a string---such as that returned by `load_file`---as input and returns another string comprising preprocessed text.

Must implement at minimum:

- Punctuation removal
- Special character handling
- Stopword removal (may be optional)

### Extract features

The preprocessed text can then be turned into feature vectors in some way, the final step before applying a machine learning algorithm to the training data.

You must implement the function `extract_features`, which takes two list of strings, a train split and a validation/test split, and returns two numerical array-like objects representing the two data splits.

### Train classifier

Implement the function that takes data produced by your feature extraction to train a classifier.

You must implement the function `train`, which takes two numerical array-like objects (representing the instances and their labels) as input and returns a trained model object. The model object must have a `predict` method that takes as input a numerical array-like object (representing instances) and returns a numerical array-like object (representing predicted labels).

### Evaluate trained classifier

Implement the function `evaluate`, which takes as input a trained model, as well as a labeled dataset, and returns the following evaluated performance metrics: recall, precision, F1, and accuracy.
