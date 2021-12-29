#SOURCE CODE
#Importing libraries and dataset
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
data = pd.read_csv("Language Detection.csv")
#Count the value count for each language.
data["Language"].value_counts()
#Separating Independent and Dependent features
X = data["Text"]
y = data["Language"]
#Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
#Text preprocessing
data_list = []
for text in X:
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        text = text.lower()
        data_list.append(text)
#Creating Bag of Words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
X.shape
#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y,
test_size = 0.20)
#Model training and prediction
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#Model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix,
classification_report
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy is :",ac)
#Confusion matrix using Seaborn heatmap
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()
#Predicting languages
def predict(text):
     x = cv.transform([text]).toarray() # converting text to
bag of words model (Vector)
     lang = model.predict(x)
     lang = le.inverse_transform(lang)
     print("The langauge is in",lang[0])
