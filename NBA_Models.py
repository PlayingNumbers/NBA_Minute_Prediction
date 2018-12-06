# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 09:31:16 2018

@author: Ken
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

def buildTS(path):
    df = pd.read_csv(path)
    df = df[['gmDate', 'playDispNm','teamAbbr','teamDayOff','playPos','playStat','playMin']]
    datfrm = datfrm = pd.DataFrame()
    for i in df.playDispNm.unique():
        pdf = df[df.playDispNm == i].sort_values(by = 'gmDate')
        pdf['prevgm'] = pdf.playMin.shift(1)
        pdf['pavg3'] = pd.rolling_mean(pdf.playMin,3)
        pdf['pavg5'] = pd.rolling_mean(pdf.playMin,5)
        pdf['pavg10'] = pd.rolling_mean(pdf.playMin,10)
        #pdf['pavg20'] = pd.rolling_mean(pdf.playMin,20)
        pdf['pmed3'] = pd.rolling_median(pdf.playMin,3)
        pdf['pmed5'] = pd.rolling_median(pdf.playMin,5)
        pdf['pmed10'] = pd.rolling_median(pdf.playMin,10)
        #pdf['pmed20'] = pd.rolling_median(pdf.playMin,20)
        pdf['pstd3'] = pd.rolling_std(pdf.playMin,3)
        pdf['pstd5'] = pd.rolling_std(pdf.playMin,5)
        pdf['pstd10'] = pd.rolling_std(pdf.playMin,10)
        #pdf['pstd20'] = pd.rolling_std(pdf.playMin,20)
        #print(pdf.tail)
        datfrm = datfrm.append(pdf.dropna())
        #print(len(datfrm))
    return datfrm

def testMetrics(dfts):
    rmseprev = sqrt(mean_squared_error(dfts.playMin, dfts.prevgm))
    rmsep3 = sqrt(mean_squared_error(dfts.playMin, dfts.pavg3))
    rmsep5 = sqrt(mean_squared_error(dfts.playMin, dfts.pavg5))
    rmsep10 = sqrt(mean_squared_error(dfts.playMin, dfts.pavg10))
    
    rmsep3med = sqrt(mean_squared_error(dfts.playMin, dfts.pmed3))
    rmsep5med = sqrt(mean_squared_error(dfts.playMin, dfts.pmed5))
    rmsep10med = sqrt(mean_squared_error(dfts.playMin, dfts.pmed10))
    
    print("Just Previous Game Prediction Score: ", rmseprev)
    print("Previous Three Games Rolling Avg. Score: ", rmsep3)
    print("Previous Five Games Rolling Avg. Score: ", rmsep5)
    print("Previous Ten Games Rolling Avg. Score: ", rmsep10)
    print("Previous Three Games Rolling Median Score: ", rmsep3med)
    print("Previous Five Games Rolling Median Score: ", rmsep5med)
    print("Previous Ten Games Rolling Median Score: ", rmsep10med)
    
def build_TrainTest(df):
    Y = df.playMin
    X = pd.get_dummies(df.drop(['gmDate','playDispNm','teamAbbr','playMin'], axis =1))
    x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size =.25, random_state =1)
    return x_train,x_test,y_train,y_test
    
def RunLinearModel(trainx,trainy,testx,testy):
    lm = linear_model.LinearRegression()
    lm.fit(trainx,trainy)
    yhat = lm.predict(testx)
    print("RMSE = ",sqrt(mean_squared_error(testy,yhat)))
    return lm, yhat

def randomForest(trainx,trainy,testx,testy):
    rf2 = RandomForestRegressor(max_features = 'log2',max_depth =20,n_estimators = 200, random_state =2)
    rf2.fit(trainx,trainy)
    features = pd.Series(rf2.feature_importances_, index = list(trainx.columns)).sort_values(ascending = False)
    yhatrf2 = rf2.predict(testx)
    print("RMSE = ",sqrt(mean_squared_error(testy,yhatrf2)))
    return rf2, yhatrf2, features
