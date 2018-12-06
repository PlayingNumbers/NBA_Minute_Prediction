# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:56:12 2018

@author: Ken
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('2012-18_playerBoxScore.csv')
df.head()
df.columns

dftrim = df[['gmDate','teamAbbr','teamDayOff', 'offLNm1',
       'offFNm1', 'offLNm2', 'offFNm2', 'offLNm3', 'offFNm3', 'playDispNm',
       'playStat', 'playMin', 'playPos', 'playHeight', 'playWeight',
       'playBDate', 'playPTS', 'playAST', 'playTO', 'playSTL', 'playBLK',
       'playPF', 'playFGA', 'playFGM', 'playFG%', 'play2PA', 'play2PM',
       'play2P%', 'play3PA', 'play3PM', 'play3P%', 'playFTA', 'playFTM',
       'playFT%', 'playORB', 'playDRB', 'playTRB', 'opptAbbr', 'opptConf',
       'opptDiv', 'opptLoc', 'opptRslt', 'opptDayOff']]

x = dftrim.head()

PTStats = df[['gmDate', 'playDispNm','teamAbbr','teamDayOff','playPos','playStat','playMin']]
sort_stat  = PTStats.sort_values(by = ['playDispNm','gmDate'])

df['playDispNm'].value_counts()

pj_tuck = sort_stat[sort_stat['playDispNm'] == 'P.J. Tucker']
pj_tuck['prevgm'] = pj_tuck.playMin.shift(1)
pj_tuck['prev5']  = pd.rolling_mean(pj_tuck.playMin,5)
pj_tuck['prev10']  = pd.rolling_mean(pj_tuck.playMin,10)
pj_tuck['prev20']  = pd.rolling_mean(pj_tuck.playMin,20)
pj_tuck['agall'] = (pj_tuck.prevgm+pj_tuck.prev5+pj_tuck.prev10+pj_tuck.prev20)/4
pj_tuck['agavgs'] = (pj_tuck.prev5+pj_tuck.prev10+pj_tuck.prev20)/3
pj_tuck['agavgs2'] = (pj_tuck.prev5+pj_tuck.prev10)/2

pj_tuck.dropna(inplace = True)

list(df.playDispNm.unique())

sort_stat.corr()
f, ax = plt.subplots(figsize=(20,18))
sns.heatmap(sort_stat.corr(), vmax=.8, square=True)

def buildTS(df):
    datfrm = pd.DataFrame()
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
        print(len(datfrm))
    return datfrm

dfts = buildTS(PTStats)
from sklearn.metrics import mean_squared_error
from math import sqrt
  
rmsprev = sqrt(mean_squared_error(dfts.playMin, dfts.prevgm))
rmsp3 = sqrt(mean_squared_error(dfts.playMin, dfts.pavg3))
rmsp5 = sqrt(mean_squared_error(dfts.playMin, dfts.pavg5))
rmsp10 = sqrt(mean_squared_error(dfts.playMin, dfts.pavg10))

rmsp3med = sqrt(mean_squared_error(dfts.playMin, dfts.pmed3))
rmsp5med = sqrt(mean_squared_error(dfts.playMin, dfts.pmed5))
rmsp10med = sqrt(mean_squared_error(dfts.playMin, dfts.pmed10))

Y = dfts.playMin
X = pd.get_dummies(dfts.drop(['gmDate','playDispNm','teamAbbr','playMin'], axis =1))



###############################################################################################
#Linear Model
###############################################################################################

from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats

X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())


x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size =.25, random_state =1)

lm = linear_model.LinearRegression()
lm.fit(x_train,y_train)


yhat = lm.predict(x_test)

sqrt(mean_squared_error(y_test,yhat))
sqrt(mean_squared_error(y_test, x_test.pavg3))
 

###############################################################################################
#RF model
###############################################################################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
rf = RandomForestRegressor()
rf.fit(x_train,y_train)
feature_imp = pd.Series(rf.feature_importances_, index = list(X.columns)).sort_values(ascending = False)
yhatrf = rf.predict(x_test)
sqrt(mean_squared_error(y_test,yhatrf))

param_grid = { 
    'n_estimators': [10,100,200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [2,5,10,20]
}

CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
CV_rfc.fit(x_train,y_train)
print(CV_rfc.best_estimator_)

rf2 = RandomForestRegressor(max_features = 'log2',max_depth =20,n_estimators = 200)
rf2.fit(x_train,y_train)

yhatrf2 = rf2.predict(x_test)

sqrt(mean_squared_error(y_test,yhatrf2))

ensm = (yhatrf2+yhat)/2
sqrt(mean_squared_error(y_test,ensm))



#################################################################################################
#Visual
#################################################################################################


plt.plot(y_test)
plt
from sklearn.metrics import mean_squared_error
from math import sqrt

rmsprev = sqrt(mean_squared_error(pj_tuck.playMin, pj_tuck.prevgm))
rmsp5 = sqrt(mean_squared_error(pj_tuck.playMin, pj_tuck.prev5))
rmsp10 = sqrt(mean_squared_error(pj_tuck.playMin, pj_tuck.prev10))
rmsp20 = sqrt(mean_squared_error(pj_tuck.playMin, pj_tuck.prev20))
rmsagg = sqrt(mean_squared_error(pj_tuck.playMin, pj_tuck.agall))
rmsagg3 = sqrt(mean_squared_error(pj_tuck.playMin, pj_tuck.agavgs))
rmsagg2 = sqrt(mean_squared_error(pj_tuck.playMin, pj_tuck.agavgs2))

from statsmodels.tsa.arima_model import ARIMA
modfit = pj_tuck[['gmDate','playMin']].values
model = ARIMA(pj_tuck.playMin.astype(float).values, order = (5,1,0))
model_fit = model.fit(disp = 0)
model_fit.summary()

yout = []
history = [x for x in pj_tuck.playMin.astype(float).values]
for i in range(len(pj_tuck.values)):
    model = ARIMA(history, order = (5,1,0))
    model_fit = model.fit(disp = 0)
    output = model_fit.forecast()
    yhat = output[0]
    yout.append(yhat)
    obs = pj_tuck.playMin.astype(float).values[i]
    history.append(obs)

pj_tuck['ARIMA'] = np.array(yout)

rmsearima = sqrt(mean_squared_error(pj_tuck.playMin, pj_tuck.ARIMA))
#set seasons --> 
#Year
# Previous Season AVG
#AVG THru Current Game 
#LAST 5 Games 
#Last 3 Games
#Last Game
#Injury 
#Spread 
#Pace/ #Possessions
#Position 
#Home/ Away