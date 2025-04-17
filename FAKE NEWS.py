import numpy as np
import streamlit as st 
import pandas as pd 
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#DATA PREPROCESSINg

news_dataset=pd.read_csv('train.csv') 
 
news_dataset.isnull().sum()
news_dataset=news_dataset.fillna('')
news_dataset['content']= news_dataset['author'] + ' ' + news_dataset['title']
X=news_dataset.drop(columns='label',axis=1)
Y=news_dataset['label']

stemmer = PorterStemmer()
 
def stemming(content):
  stemmed_content= re.sub('[^a-zA-Z)]',' ',content)
  stemmed_content=stemmed_content.lower()
  stemmed_content=stemmed_content.split()
  stemmed_content= [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content=' '.join(stemmed_content)
  return stemmed_content

news_dataset['content']=news_dataset['content'].apply(stemming)

X = news_dataset['content'].values
Y = news_dataset['label'].values

vectorizer=TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,stratify=Y,random_state=3)

model = LogisticRegression()

model.fit(X_train,y_train)
 
#WEBSITE

st.title('Fake News Detector')
input_text=st.text_input('Enter News Article')

def prediction(input_text):
    input_data= vectorizer.transform([input_text])
    prediction=model.prediction(input_data)
    return prediction[0]

if input_text:
    pred=prediction(input_text)
    if pred==1:
        st.write("This is Fake News")
     
    else:
        st.write("This is Real News")    
