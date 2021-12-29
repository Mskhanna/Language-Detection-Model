# Language-Detection-Model-ML and NLP
This project **Language Detection Model** is the task of determining the natural language that a document is written in. Recognizing text in a specific language comes naturally to a human reader familiar with the language. A trained linguist may be familiar with many dozens, but most of us will have, at some point, encountered written texts in languages they cannot place. Research into Language identification aims to mimic this human ability to recognize specific languages. It has a wide variety of applications. I am using a [Language dataset](https://www.kaggle.com/basilb2s/language-detection) from Kaggle, which contains text details for **17 different languages**. This dataset was created by scraping Wikipedia, so it contains many unwanted symbols, numbers which will affect the quality of our model. This is a solution for many artificial intelligence applications and computational linguists.



### The languages included in the dataset are given below:
1) English
2) Malayalam 
3) Hindi
4) Tamil
5) Kannada 
6) French
7) Spanish
8) Portuguese
9) Italian
10) Russian 
11) Sweedish 
12) Dutch 
13) Arabic 
14) Turkish 
15) German 
16) Danish 
17) Greek

![Dirgram](https://lh3.googleusercontent.com/Kul1_bKbt5832ppvMiTUwz1rI8O-NyShY3oGUBcbbwi-7LW8HtqagoMWjPHB3GJ4o8i4qJfL8BdazQIUbMS1gJyZKSeudMeVepSLPMl0f5Fj7upnqan5I1o-rhaVJEaOLiPT8BfAnvLLfJHIAGHdI2mDevK026QlumahgsUuB0vtpYSgeAY4xMGrcqb4CoccfQstD2_sODfuobIqQOpy3-rJXO8wl4eXXsbWPxx-wXoHSxmJ_GZiVEVvHveKdImpHkUmuOHSvlO9WHl065BFxw0l87tfWbzgGA42v3HnQ7nF5F_dtp707yqv35BmFsZE-taS5ixkuGG6NXbUvbb_NTGwz9k3iXQeYvUkzUPKlQ3kQMmGhKXBRNOKIxdZAMxnoeiuLIkqN6L4pA38xghoAStsbv5Oc_VxE7vJsa91zuL7lBoVx8R5QgXXJRXal6OOSKB8vgBQIf05Kyt-ldX6zalR6uE-RfbVyYUeNX2Au9vWTqQrhZHKJDfWLB05JyBafL9z_SNqU5A2Q8Y2Q2u5c-ScxWfRX0JGMFdXforF74qiIG-5ZgLRjEjlY5AoXdl0rMska-rJc29r7hJVbhDJXhDbWZoC7qVHASo0grZk4h_hR5AO950CF9bgEoqn_2uUNEcnVhN1X7q2pIVVkRZMMQYWdjfGAB5rDYHO6j3c3Cp9AR-D62OngGaPmaqG5FHq5M7rUDZLttWnAWYUfz8v-4XH=w976-h540-no?authuser=0/200/200)



### Table of Contents :
```
  1. Importing the required libraries
  2. Loading Dataset
  3. Preprocessing and Cleaning Data
  4. Train-Test Split
  5. Model Creation
  6. Making Predictions
```

# 1. Importing the required libraries:
```python
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.simplefilter("ignore")
``` 
# 2. Loading Dataset:
Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.
Pandas is used to read the dataset.
```python
data = pd.read_csv("Language Detection.csv")
data.head(10)
```

# 3. Preprocessing and Cleaning Data:
The data needs to be cleaned by in these two important factors:
- All special characters and numbers are removed
- Entire text is converted into lowercase
```python
       # removing the symbols and numbers
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        # converting the text to lower case
        text = text.lower()
```

# 4. Train-Test Split:
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
```

# 5. Model Creation:
I am using the Naive Bayes Algorithm for the model. You can see more about Naive Bayes [here.](https://scikit-learn.org/stable/modules/naive_bayes.html)
```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)
```

# 6. Making Predictions:
```python
predict("Hi, my name is Manmeet Singh Khanna")
```
> *Output: This langauge is English*

```python
predict("मुझे डेटा साइंस पसंद है")
```
> *Output: This langauge is Hindi*

```python
predict("Quelle belle journée")
```
> *Output: This langauge is French*


