import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

#load in the data
data = load_iris()

#convert to a dataframe
df = pd.DataFrame(data.data, columns = data.feature_names)

#create the species column
df['Species'] = data.target

#replace this with the actual names
target = np.unique(data.target)
target_names = np.unique(data.target_names)
targets = dict(zip(target, target_names))
df['Species'] = df['Species'].replace(targets)

#extract features and target variables
x = df.drop(columns="Species")
y = df["Species"]#save the feature name and target variables
feature_names = x.columns
labels = y.unique()#split the dataset
from sklearn.model_selection import train_test_split
X_train, test_x, y_train, test_lab = train_test_split(x,y,
                                                 test_size = 0.4,
                                                 random_state = 42)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth =5, random_state = 42)
clf.fit(X_train, y_train)

#import relevant packages
from sklearn import tree
import matplotlib.pyplot as plt#plt the figure, setting a black background
plt.figure(figsize=(30,10), facecolor ='k')#create the tree plot
a = tree.plot_tree(clf,
                   #use the feature names stored
                   feature_names = feature_names,
                   #use the class names stored
                   class_names = labels,
                   rounded = True,
                   filled = True,
                   fontsize=14)#show the plot
plt.show()
