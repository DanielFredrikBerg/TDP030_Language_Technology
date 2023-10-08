import nltk, time
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score
from functools import reduce
import re



def clean_review_info(text):
    text = re.sub(r'[0-9]+', '', text)
    repls = ('.', ' '), ('did n\'t', 'didn\'t'), ('wo n\'t', 'won\'t'), ('do n\'t', 'don\'t'), ('n\'t', ''), ('*', ''), (',', ' '), ('\'', ' '), ('-', ' ')
    return reduce(lambda a, kv: a.replace(*kv), repls, text)

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

df = pd.read_csv('tripadvisor_hotel_reviews.csv')
#df = pd.read_csv('Musical_instruments_reviews')


def get_sentiment_label(review):
    sentiment_scores = sia.polarity_scores(review)
    if sentiment_scores['compound'] >= 0.05:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

    
df['Review'] = df['Review'].apply(clean_review_info)
pred_start = time.perf_counter()
df['sentiment_label'] = df['Review'].apply(get_sentiment_label)
pred_end = time.perf_counter()

# Prepare the data for accuracy score
y_true = []
y_pred = []
for i, row in df.iterrows():
    rating = row['Rating']
    sentiment_label = row['sentiment_label']
    if rating == None:
        continue
    elif rating <= 2:
        y_true.append('negative')
    elif rating >= 4:
        y_true.append('positive')
    else:
        y_true.append('neutral')
    y_pred.append(sentiment_label)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate the classification report
report = classification_report(y_true, y_pred)
print(f'Performance:')
print(f'- Accuracy: {accuracy:.2f}')
print(f'- Vader Prediction Time: {pred_end - pred_start:.2f}s')
print(f'- Classification Report:\n')
print(report)

#Performance:
#- Accuracy: 0.80
#- Vader Prediction Time: 23.45s
#- Classification Report:
#
#              precision    recall  f1-score   support
#
#    negative       0.85      0.40      0.54      3214
#     neutral       0.15      0.01      0.01      2184
#    positive       0.79      0.99      0.88     15093
#
#    accuracy                           0.80     20491
#   macro avg       0.60      0.47      0.48     20491
#weighted avg       0.73      0.80      0.74     20491
