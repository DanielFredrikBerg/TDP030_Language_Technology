from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# TODO get this SVM-shit to work with other review data..
# TODO2 get Decision Trees to work at all using encoding.
# https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931
# TODO3 get Random Forest to work.
# TODO4 preprocess and save the cleaned data somewhere.
# TODO5 get cross comparison to work with all models..


reviews = pd.read_csv("Musical_instruments_reviews.csv")
#fine_food = pd.read_csv("amazon_fine_food.csv")

category_column = "reviewText"
target_column = "overall"

# reviews.Rating.value_counts().plot(kind='bar')
# plt.show()


#print(reviews.Rating.value_counts())

count_score_5, count_score_4, count_score_3, count_score_2, count_score_1 = reviews.Rating.value_counts()

class_s1 = reviews[reviews[target_column] == 1]
class_s2 = reviews[reviews[target_column] == 2]
class_s3 = reviews[reviews[target_column] == 3]
class_s4 = reviews[reviews[target_column] == 4]
class_s5 = reviews[reviews[target_column] == 5]

################################### UNDER SAMPLING

class_s2_under = class_s2.sample(count_score_1)
class_s3_under = class_s3.sample(count_score_1)
class_s4_under = class_s4.sample(count_score_1)
class_s5_under = class_s5.sample(count_score_1)

test_under_sampled = pd.concat([class_s1, class_s2_under, class_s3_under, class_s4_under, class_s5_under], axis=0)
#test_under_sampled.Rating.value_counts().plot(kind='bar')
#plt.show()

#print(test_under_sampled)

################################### OVER SAMPLING

class_s1_over = class_s1.sample(count_score_5, replace=True)
class_s2_over = class_s2.sample(count_score_5, replace=True)
class_s3_over = class_s3.sample(count_score_5, replace=True)
class_s4_over = class_s4.sample(count_score_5, replace=True)

test_over_sampled = pd.concat([class_s1_over, class_s2_over, class_s3_over, class_s4_over, class_s5], axis=0)
# test_over_sampled.Rating.value_counts().plot(kind='bar')
# plt.show()


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from numpy import asarray
from sklearn.preprocessing import OneHotEncoder


data, target = reviews[category_column], reviews[target_column]
#data, target = test_under_sampled[category_column], test_under_sampled[target_column]
#data, target = test_over_sampled[category_column], test_over_sampled[target_column]

#data_train, data_test, target_train, target_test
X_train, X_test, y_train, y_test = train_test_split(
    data, target, random_state=1337)

# oneshot_encoder = OneHotEncoder(sparse_output=False)
# oneshot_encoder.fit([X_train])
# X_train = oneshot_encoder.transform([X_train])
#X_test = oneshot_encoder.transform([X_test])


# # ordinal encode target variable
# label_encoder = LabelEncoder()
# label_encoder.fit(y_train)
# y_train = label_encoder.transform(y_train)
# y_test = label_encoder.transform(y_test)

# clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))

# ff_data, ff_target = fine_food["Text"], fine_food["Score"]

# ff_X_train, ff_X_test, ff_y_train, ff_y_test = train_test_split(
#     ff_data, ff_target, random_state=1337)

vectorizer = CountVectorizer()
svm = LinearSVC(max_iter=10000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
_ = svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))


# Normal Results
#               precision    recall  f1-score   support

#            1       0.62      0.57      0.59       364
#            2       0.39      0.29      0.33       465
#            3       0.28      0.29      0.29       524
#            4       0.44      0.45      0.44      1498
#            5       0.68      0.72      0.70      2272

#     accuracy                           0.54      5123
#    macro avg       0.48      0.46      0.47      5123
# weighted avg       0.54      0.54      0.54      5123



# Random Under Sampled results
#               precision    recall  f1-score   support

#            1       0.61      0.62      0.62       337
#            2       0.37      0.38      0.38       359
#            3       0.40      0.39      0.39       347
#            4       0.39      0.38      0.38       360
#            5       0.60      0.60      0.60       374

#     accuracy                           0.47      1777
#    macro avg       0.47      0.48      0.47      1777
# weighted avg       0.47      0.47      0.47      1777


# Over sample results (overfit probably)
#               precision    recall  f1-score   support

#            1       0.98      1.00      0.99      2224
#            2       0.97      0.98      0.98      2273
#            3       0.91      0.97      0.94      2238
#            4       0.77      0.81      0.79      2303
#            5       0.85      0.72      0.78      2280

#     accuracy                           0.90     11318
#    macro avg       0.90      0.90      0.90     11318
# weighted avg       0.89      0.90      0.89     11318
