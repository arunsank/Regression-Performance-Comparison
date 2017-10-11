# -*- coding: utf-8 -*-
"""
Created on Sat May  6 02:37:06 2017

@author: Arun Ram
"""

import os
import time
import json
import keras
import pandas as pd
import numpy as np
import statsmodels.formula.api as sfa
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn import preprocessing
from pandas.tools.plotting import scatter_matrix
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import KernelPCA
#intitalize neural network
from keras.models import Sequential
#create ANN layers
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV



def visualize(df):
    """
    Visualize dataset
    """
    featurenames= ['x0','x1','x2','x3','x4','x5','x6','x7','x8','y']
    #Histogram
    hist= df.hist(featurenames)
    #Boxplot
    df.plot(kind='box',subplots=True, layout=(4,3),sharex=False, sharey= False,title= 'Box Plots')
    #density plots
    dens = df.plot(kind='density',subplots=True,layout=(4,3),sharex=False,title= 'Density Plots')
    #Correlation
    sns.heatmap(df.corr())
    #Scatter matrix
    scatter_matrix(df)
    plt.show()
    

def linearity(regressor,xtrain,ytrain):
    """
    Check for SVR hyper parameter optimization
    """
    start = time.time()
    params = [{'C': [2000], 'kernel': ['linear']},
              {'C': [2000], 'kernel': ['rbf'], 'gamma': [0.01]},
               {'C': [2000], 'kernel': ['poly'], 'gamma': [0.01]}]
    
    gsearch = GridSearchCV(estimator= regressor , param_grid = params, scoring = 'r2',
                       cv= 3, n_jobs=-1)
    gsearch = gsearch.fit(xtrain,ytrain)
    best_accuracy = gsearch.best_score_
    best_parameters = gsearch.best_params_
    end =time.time()
              
    return best_accuracy,best_parameters

def rfparametertuning(regressor,xtrain,ytrain):
    """
    Check for RF hyper parameter optimization
    """
    param_grid = { 
    'n_estimators': [200,400,900],
    'max_features': ['auto', 'sqrt', 'log2']
    }
    
    gsearch = GridSearchCV(estimator=regressor, param_grid=param_grid, cv= 3)
    
    gsearch= gsearch.fit(xtrain,ytrain)
    best_score = gsearch.best_score_
    best_parameters = gsearch.best_params_
    return best_score,best_parameters

def SVRegress(xtrain,ytrain,xtest,ytest,Cparam,gamm,kern,x,y):
    
    """
    SVR regression model
    """
    start =time.time()    
    SVregression = SVR(C=Cparam, gamma= gamm,kernel= kern)
    SVregression.fit(xtrain,ytrain)
    y_pred= SVregression.predict(xtest)
    mse= ((ytest-y_pred)**2).mean()
    rsq= r2_score(ytest,y_pred)
    n= len(xtest)
    p= xtest.shape[1]
    adj = 1- ((1-rsq**2) * ((n-1)/ (n-p-1)))
    scores = cross_val_score(SVregression, x, y, cv=10)
    end =time.time()
    return mse,rsq,adj,(end-start),scores

def rforest(xtrain,ytrain,xtest,ytest,features,estimators,x,y):
    """
    Random forest model- ensemble learning

    """
    
    start =time.time()
    forest = RandomForestRegressor(max_features=features, n_estimators= estimators, random_state=0)
    forest= forest.fit(xtrain,ytrain)
    y_pred= forest.predict(xtest)
    mse= ((ytest-y_pred)**2).mean()
    rsq= r2_score(ytest,y_pred)
    n= len(xtest)
    p= xtest.shape[1]
    adj = 1- ((1-rsq**2) * ((n-1)/ (n-p-1)))
    scores = cross_val_score(forest, x, y, cv=10)
    end =time.time()
    return mse,rsq,adj,(end-start),scores


def ANN(xtrain,ytrain,xtest,ytest,x,y):
    """
    ANN model
    Parameters used:
    out_dim - number of nodes in input layer
    init - Initialize weights to small numbers close to 0
    (Glorot with weights sampled from the Uniform distribution)
    activation - rectifier function , relu 
    input_dim - number of input or indepedant features
    """
    
    
    start =time.time()    
    #Model start
    regressor = Sequential()
    #features = xtrain.shape[1]
    #ytrain = ytrain.reshape((len(ytrain)),1)
    
    #first hidden layer
    regressor.add(Dense(output_dim=9,init='glorot_uniform', activation='relu',input_dim=9))
    #uniform
    #second hidden layer    
    regressor.add(Dense(output_dim=9,init='glorot_uniform', activation='relu'))
    
    #third hidden layer
    regressor.add(Dense(output_dim=9,init='glorot_uniform', activation='relu'))

    #output layer
    regressor.add(Dense(output_dim=1,init='glorot_uniform', activation='relu'))
    regressor.compile(optimizer= 'adam',loss= 'mean_squared_error')
    regressor.fit(xtrain,ytrain, batch_size= 10, nb_epoch=500)
    ypred= regressor.predict(xtest,batch_size=10)
    rsq= r2_score(ytest,ypred)
    score = regressor.evaluate(xtest, ytest, batch_size=10)
    n= len(xtest)
    p= xtest.shape[1]
    adj = 1- ((1-rsq**2) * ((n-1)/ (n-p-1)))
    print("Results:",score )
    print ("Rsquared:", rsq)
    print ("Adjusted R squared:", adj)    
    end =time.time()
    return score,rsq,adj,(end-start)
    
    """0.984034391008
Adjusted R squared: 0.967502106733"""

def ANNCV(x,y):
    """
    K fold Cross validation on our ANN model
    """
    kf = KFold(n_splits=10, shuffle=False)
    
    scores= []
    rsquar= []
    adj= []
    for training,testing in kf.split(x,y):
        regressor = Sequential()
        #first hidden layer
        regressor.add(Dense(output_dim=9,init='glorot_uniform', activation='relu',input_dim=9))
        #uniform
        #second hidden layer    
        regressor.add(Dense(output_dim=9,init='glorot_uniform', activation='relu'))
        
        #third hidden layer
        regressor.add(Dense(output_dim=9,init='glorot_uniform', activation='relu'))

        #output layer
        regressor.add(Dense(output_dim=1,init='glorot_uniform', activation='relu'))
    
        regressor.compile(optimizer= 'adam',loss= 'mean_squared_error')
        
        regressor.fit(x[training],y[training], batch_size= 10, nb_epoch=500)
    
        scoreval = regressor.evaluate(x[testing], y[testing], batch_size= 10)
        scores.append(scoreval)
        ypred= regressor.predict(x[testing],batch_size=10)
        rsq= r2_score(y[testing],ypred)
        if(rsq>0):
            rsquar.append(rsq)
        n= len(x[testing])
        p= x[testing].shape[1]
        adjs = 1- ((1-rsq**2) * ((n-1)/ (n-p-1)))
        adj.append(adjs)
    
    return scores,rsquar,adj


def featureselection(X):
    """
    Operations :
    Feature selection using Backward elimination process
    significance level p-value >0.05 
    """
    count= len(X)
    X= np.append(arr=np.ones((count,1)).astype(int), values= X, axis=1) 
    X_optimal = X[:,list(range(0,X.shape[1]))]
    Ols = sfa.OLS(endog=y,exog= X_optimal).fit()
    Ols.summary()

    #Compare with p-values , find predictor with highest p-value if p-value > 0.05 then we continue
    #Significance Level =0.05

    elem_list = list(range(0,X.shape[1]))

    #first elimination

    elem_list.pop(10)
    X_optimal =X[:,elem_list]

    Ols = sfa.OLS(endog=y,exog=X_optimal).fit()

    Ols.summary()

    #second elimination
    elem_list.pop(3)
    X_optimal =X[:,elem_list]
    Ols = sfa.OLS(endog=y,exog=X_optimal).fit()
    Ols.summary()

    #Third elimination
    elem_list.pop(10)
    X_optimal =X[:,elem_list]
    Ols = sfa.OLS(endog=y,exog=X_optimal).fit()
    Ols.summary()
    #Export and analyze
    X= X_optimal[:,1:X_optimal.shape[1]]
    return X

def preprocess(dataset):
    """
    Operations:
    1. Check for nulls
    2. Impute if nulls are present based on strategy
    3. Split dependant and independant variables
    4. return x and y
    """
    #check for null columns
    print('Checking for nulls')
    nullcount= dataset.isnull().sum().sum()
    print("Number of nulls=", nullcount)
    featurecount = len (dataset.columns)
    
    if nullcount:
        
        #mean  - Strategy -mean, median,most frequent
        imputer= Imputer(missing_values='NaN',strategy="mean",axis=0)
        x= dataset.iloc[:,1:-1].values
        y= dataset.iloc[:,featurecount-1].values
    
        #fit imputer
        imputer.fit(x[:,0:11])
        x[:,0:11] = imputer.transform(x[:,0:11])
        return x,y
    else:

        x= dataset.iloc[:,1:-1].values
        y= dataset.iloc[:,featurecount-1].values
    
        return x,y
        

def scalerfunc(x):
    """
    Scales all indepedant variables to a range of 0 -1, so that computational
    efficiency is improved for our model
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scal = min_max_scaler.fit_transform(x)
    return x_scal


def resultviz(regressor,x,y):
    """
    Visualizes decision boundaries of our models
    
    
    """
    pca = KernelPCA(n_components = 2, kernel = 'rbf')
    X_pca = pca.fit_transform(x)
    regressor.fit(X_pca, y)
    cm = plt.get_cmap('jet')
    X_set, y_set = X_pca, y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = cm)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c = cm(i), label = j, alpha = 0.5)
    plt.title('Regression decision boundary')
    plt.xlabel('PC one')
    plt.ylabel('PC two')
    plt.show()





if __name__  == "__main__":
    print("Start of main")
    print("Import data")
    directory= os.getcwd()
    os.chdir(directory)
    dataset = pd.read_csv('data.csv')
    
    #Step 1 - preprocess
    x,y= preprocess(dataset)
    print("End of preprocessing")
    
    #Step 2- Feature selection - Backward elimination
    print("Start of feature selection and feature scaling")
   
    x =featureselection(x)
    
    #Step 3 - Feature scaling so that all features are in the same scale - MinMaxScaler is used 
    x_scal = scalerfunc(x)
    
    print("End of feature selection and feature scaling")
    #Step 4 - Visualize to get better intuition
    
    print("Start of Visualization")
    df = pd.DataFrame({'x0':x[:,0],'x1':x[:,1],'x2':x[:,2],'x3':x[:,3],'x4':x[:,4],
                       'x5':x[:,5],'x6':x[:,6],'x7':x[:,7],'x8':x[:,8],'y':y})
    df.to_csv('featureselected.csv',sep=',')
    visualize(df)
    print("End of Visualization")
    
    #Step 5 : Train , test split
    print("Start of Implementation")
    
    x_train,x_test,ytrain,ytest = train_test_split(x,y,test_size=0.2,train_size=0.8, random_state= 0 )
    
    #Scaling - Features scaled to a scale of 0 to 1
    xtrain = scalerfunc(x_train)
    xtest = scalerfunc(x_test)
    
    #Step 6 - Regression Algorithms evaluation
    
    print("Start of Random forest regression")
    """
    Randomforestregressor - enseble learning 
    output: mean prediction of individual trees
  
    """
    #Step1
    #Hyper parameter tuning
    regressor = RandomForestRegressor()
    best_score,best_parameters= rfparametertuning(regressor,xtrain,ytrain)
    print (" Best Score:", best_score)
    print ("The best parameters for RandomForestRegressor():", best_parameters)
    #Step2
    #Random Forest Regressor    
    mse,rsq,adj,tim,score=rforest(xtrain,ytrain,xtest,ytest,best_parameters['max_features'],best_parameters['n_estimators'],x,y)
    print (" Random Forest regressor:")    
    print ("Mean squared error:", mse)
    print ("R-squared:", rsq)
    print ("Adjusted R-squared:", adj)
    print ("Execution time:", tim)
    print ("Cross Validation score:", score.mean())
    
    regress = RandomForestRegressor(max_features=best_parameters['max_features'], n_estimators= best_parameters['n_estimators'], random_state=0)
    resultviz(regress,xtrain,ytrain)
    
    resultviz(regress,xtest,ytest)
    print("End of Random forest regression")

    #Step3
    #Hyper parameter tuning
    print("Start of SVR")
    regressor= SVR()
    score, params =linearity(regressor,xtrain,ytrain)
    print (" Best score:", score)
    print ("The best parameters for SVR():", params)
    
    #SVR Regressor
    mse1,rsq1,adj1,tim1,score1= SVRegress(xtrain,ytrain,xtest,ytest,params['C'],params['gamma'],params['kernel'],x,y)
    
    print ("SVR (rbf kernel) regressor:")    
    print ("Mean squared error:", mse1)
    print ("R-squared:", rsq1)
    print ("Adjusted R-squared:", adj1)
    print ("Execution time:", tim1)
    print ("Cross Validation score:", score1.mean())
    
    regress = SVR(C=params['C'], gamma= params['gamma'],kernel= params['kernel'])
    resultviz(regress,xtrain,ytrain)
    
    resultviz(regress,xtest,ytest)
    print("End of SVR")
    #Artificial Neural Network - Keras
    print("Start of ANN")
    mse2,rsq2,adj2,tim2= ANN(x_train,ytrain,x_test,ytest,x,y)
    
    sc,rs,ad = ANNCV(x,y)
    
    cvsk = np.mean(sc)
    print("End of ANN")
    #store values 
    cvr= np.mean(score)
    cvsv= np.mean(score1)
    result_dict = [{'Model':'Random Forest' ,'mse': mse, 'Rsq':rsq , 'Adj-R' :adj, 'CV' :cvr ,'Time': tim},
            {'Model':'SVR' ,'mse1': mse1, 'Rsq1':rsq1, 'Adj-R1' :adj1,'CV' : cvsv, 'Time': tim1},
            {'Model':'ANN' ,'mse1': mse2, 'Rsq1':rsq2, 'Adj-R1' :adj2,'CV' : cvsk, 'Time': tim2}]
    
    
    #output results to text
    json.dump(result_dict, open("output.txt",'w'))
    
    
    
    