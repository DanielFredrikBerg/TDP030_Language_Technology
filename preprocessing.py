import csv
from functools import reduce
import re
import pandas as pd


test_df = pd.read_csv('tripadvisor_hotel_reviews.csv')
print(test_df.head(1))
print('--'*10)


# file_ = open('tripadvisor_hotel_reviews.csv')

# csvreader = csv.reader(file_)

# header = []
# header = next(csvreader)
# print(header)


# def clean_review_info(text):
#     text = re.sub(r'[0-9]+', '', text)
#     repls = ('.', ' '), ('did n\'t', ''), ('wo n\'t', ''), ('n\'t', ''), (',', ' '), ('\'', ' '), ('-', ' ')
#     return reduce(lambda a, kv: a.replace(*kv), repls, text)
    


# rows = []
# for row in csvreader:
#     review_text = row[0]
#     review_score = row[1]
#     rows.append([ clean_review_info(review_text), review_score ])
# print(rows[2000])


# file_.close()

