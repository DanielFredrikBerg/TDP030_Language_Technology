import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the training dataset into a Pandas dataframe
test_df = pd.read_csv('tripadvisor_hotel_reviews.csv')

# Load the testing dataset into a Pandas dataframe
train_df = pd.read_csv('women_reviews.csv')

# Extract the text and target variables from the training data
train_text = train_df['Review Text']
train_target = train_df['Rating']

# Extract the text and target variables from the testing data
test_text = test_df['Review']
test_target = test_df['Rating']


# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer to the training data
tfidf_vectorizer.fit(train_text.values.astype('U'))

# Transform the training and testing data into TF-IDF features
train_features = tfidf_vectorizer.transform(train_text.values.astype('U'))
test_features = tfidf_vectorizer.transform(test_text.values.astype('U'))

# Create a Random Forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rfc.fit(train_features, train_target)

# Use the classifier to make predictions on the testing data
predictions = rfc.predict(test_features)

# Print the classification report
print("Trained on tripadvisor, tested on womens clothing reviews")
print(classification_report(test_target, predictions))
