#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from data_loader import get_data,review_to_words

reviews,y = get_data()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

X = []
for x in reviews:
    X.append(review_to_words(x))
    
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

cv=CountVectorizer(min_df=1,binary=False,ngram_range=(1,3))

cv_train_reviews=cv.fit_transform(X_train)

cv_test_reviews=cv.transform(X_test)
mnb=MultinomialNB(fit_prior=False,alpha = 0.1)

mnb_bow=mnb.fit(cv_train_reviews,y_train)
print(mnb_bow)

mnb_bow_predict=mnb.predict(cv_test_reviews)
print(mnb_bow_predict)

mnb_bow_score=accuracy_score(y_test,mnb_bow_predict)
print("mnb_bow_score :",mnb_bow_score)

mnb_bow_report=classification_report(list(le.inverse_transform(y_test)),list(le.inverse_transform(mnb_bow_predict)),target_names=['1','2','3','4','7','8','9','10'])
print(mnb_bow_report)
#cm_bow=confusion_matrix(y_test,mnb_bow_predict,labels=[1,0])
#print(cm_bow)
