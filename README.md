# TITANIC--Gender-survival-prediction
My prroject from a competition i signed up for in kaggle. Gender survival prediction. Did more female surviced than male: II used the Neural Network Algorithm to train the data.
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import tensorflow
import keras
train= pd.read_csv('train.csv')
train.head()
train.drop(train.columns[2:12], axis=1, inplace=True)
train
train.replace(('male','female'), (0,1), inplace=True)
feature_names=['PassengerId', 'Survived', 'Pclass', 'Age', 'Sex', 'Parch', 'Fare']
all_features = train[feature_names].values
all_classes =train['Sex'].values
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import cross_val_score
def create_model():
    model=Sequential()
    model.add(Dense(32, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,  kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model
from keras.wrappers.scikit_learn import KerasClassifier
estimator= KerasClassifier(build_fn=create_model, nb_epoch=100, verbose=0)
cv_scores=cross_val_score(estimator, all_features, all_classes, cv=10)
cv_score.mean()
0.49707602262496947
