# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
@author: Roshni_Mohandas
"""



import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


#Importing the dataset 
dataset = pd.read_csv('D:/Python_Codes/train.csv')
X=dataset.iloc[:,2:59].values
y=dataset.iloc[:,1].values

# Taking care of missing data 
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer = imputer.fit(X[:,0:57])
X[:,0:57]=imputer.transform(X[:,0:57])

## Encoding categorical data 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder 
labelencoder_X= LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
## Encoding independent variable 
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)


# Splitting the dataset into train and test 
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=123)


# Feature scaling 
from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

##Making ANN 
#Importing keras libraries and packages 
import keras 
# Sequential to initialise NN
# Dense module to build layers of ANN
from keras.models import Sequential 
from keras.layers import Dense


'''
ANN Steps
1) Randomly initialise weights (Close to 0, but not 0)
2) Input first observation in the dataset in the input layer , each feature in one input layer 
3) Forward propagation : From left to right ,the neurons are actiavated in a way that the impact of 
each neurons activity is limited by the weights. Propagate the activations until getting the 
predicted results y 
4) Compare the predicted resul and actual. Measure the erro
5) Back propagation from right to left . update the weights 
6) Repeat steps 1 to 5 and update the weights after each obs( Reinforcement learning)
or repeat steps 1 to 5 and update the weights after a batch of obs(Batch learning)
7) When a whole training set is passed through ANN, thats an epoch, repeat epochs
'''

#Initializing ANN ( Initialise by defining it as sequence of layers)
classifier = Sequential()
# Adding first layers of ANN
## Adding input layer and first hidden layer 
classifier.add(Dense(output_dim=29,kernel_initializer ='uniform',activation='relu',
                     input_dim=57)) # one inout and one hidden 
## Adding second hidden layer 
classifier.add(Dense(output_dim=29,kernel_initializer ='uniform',activation='relu',)) # 

#Adding output layer  , only one node in output layer 
classifier.add(Dense(output_dim=1,kernel_initializer ='uniform',activation='sigmoid',)) # 

## Compiling ANN ( Applying stochastic Gradient descent)
# For binary outcome = binary_crossentropy 
# more than two category outcome = categorical_crossentropy 
classifier.compile(optimizer='adam',loss='binary_crossentropy' , metrics=['accuracy'])

# Fit ANN to training set 
# batch size, number of observation after which you want to update weight 
# Epochs, training numbers 
 classifier.fit(X_train, y_train,batch_size=10, nb_epoch=10)


#Predicting the test results 
y_pred=classifier.predict(X_test)  # probability value outcomess
y_pred=(y_pred>0.5)

#Making the confusion matrix 
from sklearn.metrics import confusion_matrix 
cm=confusion_matrix(y_test, y_pred)

## Evaluating improving and tuning ANN 
# bias variance trade offs 
# kfold , 10 fold, train on 9 fold and test on 1 fold. different combiations
# since using keras , using a wrapper from sklearn 
from keras.wrappers.scikit_learn import KerasClassifier
 from sklearn.model_selection import cross_val_score
from keras.models import Sequential  
from keras.layers import Dropout

 #Build ANN classiifer 
def build_classifier():
    classifier.add(Dense(output_dim=29,kernel_initializer ='uniform',activation='relu',input_dim=57))
    classifier.add(Dense(output_dim=29,kernel_initializer ='uniform',activation='relu',)) 
    classifier.add(Dense(output_dim=1,kernel_initializer ='uniform',activation='sigmoid',)) 
    classifier.compile(optimizer='adam',loss='binary_crossentropy' , metrics=['accuracy'])
    return classifier 

# building new classifier 
# n_jobs number of CPUs used , -1 uses all CPUS
classifier = KerasClassifier(build_fn= build_classifier,batch_size=10, nb_epoch=10  )
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()

## Droput regularization 
# Overfitting on the training set , reduce it using regularization 
from keras.layers import Dropout
# Applying to first and second hidden layers. When there is overfitting, apply to all 
classifier.add(Dense(output_dim=29,kernel_initializer ='uniform',activation='relu',input_dim=57))
classifier.add(Dropout(p=0.1))

classifier.add(Dense(output_dim=29,kernel_initializer ='uniform',activation='relu',)) 
classifier.add(Dropout(p=0.1))

## Parameter tuning 
# Hyper parameters # epochs, batch size, number of neurons 
## Finding best values of hyperparameters, finding it using gridsearch 
from keras.wrappers.scikit_learn import KerasClassifier
 from sklearn.model_selection import GridSearchCV
 from keras.models import Sequential  
from keras.layers import Dropout
def build_classifier():
    classifier.add(Dense(output_dim=29,kernel_initializer ='uniform',activation='relu',input_dim=57))
    classifier.add(Dense(output_dim=29,kernel_initializer ='uniform',activation='relu',)) 
    classifier.add(Dense(output_dim=1,kernel_initializer ='uniform',activation='sigmoid',)) 
    classifier.compile(optimizer='adam',loss='binary_crossentropy' , metrics=['accuracy'])
    return classifier 
# building new classifier 
# n_jobs number of CPUs used , -1 uses all CPUS
classifier = KerasClassifier(build_fn= build_classifier,batch_size=10, nb_epoch=10  )
parameters = {'batch_size':[25,32],
              'nb_epochs':[100,500]
              'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator=classifier, 
                           param_grid=parameters,
                           scoring = 'accuracy',
                           cv=10)
grid_search = grid_search.fit(X_train,y_train)
best_parameters= grid_search.best_params_
best_accuracy = grid_search.best_score_



