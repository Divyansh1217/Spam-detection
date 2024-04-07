import pandas as pd
import streamlit as st
data=pd.read_csv("spam.csv",encoding='latin-1')
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
st.write(data.head())
data=data[["v1","v2"]]
x=np.array(data["v2"])
y=np.array(data["v1"])
cv=CountVectorizer()
X=cv.fit_transform(x)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
clf=MultinomialNB()
clf.fit(X_test,y_test)
sample=st.text_input("Enter the message")
data=cv.transform([sample]).toarray()
st.write(clf.predict(data))

