import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Load the reviews.csv file into a Pandas dataframe
df = pd.read_csv('tripadvisor_hotel_reviews.csv')

# Extract the review text and the corresponding scores from the dataframe
text_data = df['Review']
target = df['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_data, target, test_size=0.2, random_state=42)

# Create an instance of the TfidfVectorizer class
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer to the training data
tfidf_vectorizer.fit(X_train)

# Transform the training and testing data into matrices of TF-IDF features
X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Define the hyperparameters to tune
params = {'max_depth': Integer(5, 50),
          'min_samples_split': Integer(2, 20),
          'min_samples_leaf': Integer(1, 10)}

# Create an instance of the DecisionTreeClassifier class
clf = DecisionTreeClassifier()

# Create a BayesSearchCV object to search over the hyperparameters
bayes_search = BayesSearchCV(clf, params, cv=5, n_iter=10, n_jobs=-1)

# Fit the BayesSearchCV object to the training data
bayes_search.fit(X_train_tfidf, y_train)

# Use the best estimator to make predictions on the testing data
best_clf = bayes_search.best_estimator_
predictions = best_clf.predict(X_test_tfidf)

# Print the classification report
print(classification_report(y_test, predictions))
