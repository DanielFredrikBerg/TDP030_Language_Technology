import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from functools import reduce
import re

def clean_review_info(text):
    text = re.sub(r'[0-9]+', '', text)
    repls = ('.', ' '), ('did n\'t', 'didn\'t'), ('wo n\'t', 'won\'t'), ('do n\'t', 'don\'t'), ('n\'t', ''), ('*', ''), (',', ' '), ('\'', ' '), ('-', ' ')
    return reduce(lambda a, kv: a.replace(*kv), repls, text)

def recode_score(score):
    if score in [1, 2]:
        return 1
    elif score == 3:
        return 2
    elif score in [4, 5]:
        return 3
    
# Load the reviews.csv file into a Pandas dataframe
df = pd.read_csv('tripadvisor_hotel_reviews.csv')

# Extract the review text and the corresponding scores from the dataframe
df['Rating'] = df['Rating'].apply(recode_score)
df['Review'] = df['Review'].apply(clean_review_info)

text_data = df['Review']
target = df['Rating']



X_train, X_test, y_train, y_test = train_test_split(
    text_data, target, random_state=1337)

# Create an instance of the TfidfVectorizer class
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer to the train data
tfidf_vectorizer.fit(X_train)

# Transform the training and testing data into matrices of TF-IDF features
X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Define the hyperparameters to tune
params = {'max_depth': [5, 10, 20, None],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4]}

# Create an instance of the DecisionTreeClassifier class
clf = DecisionTreeClassifier()

# Create a GridSearchCV object to search over the hyperparameters
grid_search = GridSearchCV(clf, params, cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train_tfidf, y_train)

# Use the best estimator to make predictions on the testing data
best_clf = grid_search.best_estimator_
predictions = best_clf.predict(X_test_tfidf)
print(predictions, end='\n\n')
# Print the classification report
print(classification_report(y_test, predictions))


# Initial Decision Tree scoring on tripadvisor data
#               precision    recall  f1-score   support

#            1       0.41      0.37      0.39       364
#            2       0.21      0.18      0.19       465
#            3       0.21      0.21      0.21       524
#            4       0.36      0.37      0.36      1498
#            5       0.60      0.61      0.60      2272

#     accuracy                           0.44      5123
#    macro avg       0.36      0.35      0.35      5123
# weighted avg       0.44      0.44      0.44      5123


# Extensive Hyper parameter tuning
# precision    recall  f1-score   support

#            1       0.58      0.12      0.20       364
#            2       0.24      0.35      0.28       465
#            3       0.25      0.01      0.01       524
#            4       0.39      0.36      0.37      1498
#            5       0.57      0.75      0.65      2272

#     accuracy                           0.48      5123
#    macro avg       0.41      0.32      0.30      5123
# weighted avg       0.46      0.48      0.44      5123


# Extensive Hyper parameter tuning (3-categories)
#               precision    recall  f1-score   support

#            1       0.61      0.45      0.52       829
#            2       0.27      0.11      0.16       524
#            3       0.82      0.93      0.87      3770

#     accuracy                           0.77      5123
#    macro avg       0.56      0.50      0.52      5123
# weighted avg       0.73      0.77      0.74      5123



# Randomized Hyper parameter tuning
#               precision    recall  f1-score   support

#            1       0.41      0.42      0.41       292
#            2       0.23      0.16      0.19       333
#            3       0.27      0.15      0.19       432
#            4       0.43      0.36      0.39      1252
#            5       0.58      0.74      0.65      1790

#     accuracy                           0.49      4099
#    macro avg       0.38      0.37      0.37      4099
# weighted avg       0.46      0.49      0.47      4099


# Bayesian Optimization parameter tuning
#               precision    recall  f1-score   support

#            1       0.39      0.41      0.40       292
#            2       0.25      0.17      0.21       333
#            3       0.34      0.06      0.10       432
#            4       0.41      0.33      0.37      1252
#            5       0.58      0.80      0.67      1790

#     accuracy                           0.50      4099
#    macro avg       0.39      0.35      0.35      4099
# weighted avg       0.46      0.50      0.46      4099
