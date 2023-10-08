import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the TripAdvisor hotel reviews dataset into a Pandas dataframe
df = pd.read_csv('tripadvisor_hotel_reviews.csv')


# Define a function to recode the values of the Score column
def recode_score(score):
    if score in [1, 2]:
        return 1
    elif score == 3:
        return 2
    elif score in [4, 5]:
        return 3

# Apply the recode_score function to the Score column using the apply method
df['Rating'] = df['Rating'].apply(recode_score)


# Extract the review text and the corresponding ratings from the dataframe
text_data = df['Review']
target = df['Rating']

#print(df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_data, target, test_size=0.5, random_state=42)

# Create an instance of the TfidfVectorizer class
tfidf_vectorizer = TfidfVectorizer(
    min_df=5, max_df=0.9,
    ngram_range=(1,1),
    stop_words='english',
    use_idf=True, smooth_idf=True, sublinear_tf=True
)

# Fit the vectorizer to the training data
tfidf_vectorizer.fit(X_train.values.astype('U'))

# Transform the training and testing data into matrices of TF-IDF features
X_train_tfidf = tfidf_vectorizer.transform(X_train.values.astype('U'))
X_test_tfidf = tfidf_vectorizer.transform(X_test.values.astype('U'))

# Create an instance of the RandomForestClassifier class
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rfc.fit(X_train_tfidf, y_train)

# Use the classifier to make predictions on the testing data
predictions = rfc.predict(X_test_tfidf)

# Print the classification report
print(classification_report(y_test, predictions))


# Random Forest "out of the box"
#               precision    recall  f1-score   support

#            1       0.80      0.38      0.52       292
#            2       0.33      0.04      0.07       333
#            3       0.75      0.01      0.01       432
#            4       0.42      0.26      0.32      1252
#            5       0.54      0.95      0.69      1790

#     accuracy                           0.53      4099
#    macro avg       0.57      0.33      0.32      4099
# weighted avg       0.53      0.53      0.44      4099


# 1,3-ngram_range option
#               precision    recall  f1-score   support

#            1       0.78      0.42      0.55       292
#            2       0.33      0.02      0.03       333
#            3       0.29      0.00      0.01       432
#            4       0.42      0.28      0.34      1252
#            5       0.55      0.95      0.69      1790

#     accuracy                           0.53      4099
#    macro avg       0.47      0.33      0.32      4099
# weighted avg       0.48      0.53      0.45      4099
