import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from scipy.stats import randint

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
params = {'max_depth': randint(1, 20),
          'min_samples_split': randint(2, 10),
          'min_samples_leaf': randint(1, 4)}

# Create an instance of the DecisionTreeClassifier class
clf = DecisionTreeClassifier()

# Create a RandomizedSearchCV object to search over the hyperparameters
random_search = RandomizedSearchCV(clf, params, cv=5, n_iter=10, random_state=42)

# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train_tfidf, y_train)

# Use the best estimator to make predictions on the testing data
best_clf = random_search.best_estimator_
predictions = best_clf.predict(X_test_tfidf)

# Print the classification report
print(classification_report(y_test, predictions))
