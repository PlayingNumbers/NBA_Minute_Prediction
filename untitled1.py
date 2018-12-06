# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:56:12 2018

@author: Ken
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv('2012-18_playerBoxScore.csv')
df.head()
df.columns

dftrim = [['gmDate','teamAbbr','teamDayOff', 'offLNm1',
       'offFNm1', 'offLNm2', 'offFNm2', 'offLNm3', 'offFNm3', 'playDispNm',
       'playStat', 'playMin', 'playPos', 'playHeight', 'playWeight',
       'playBDate', 'playPTS', 'playAST', 'playTO', 'playSTL', 'playBLK',
       'playPF', 'playFGA', 'playFGM', 'playFG%', 'play2PA', 'play2PM',
       'play2P%', 'play3PA', 'play3PM', 'play3P%', 'playFTA', 'playFTM',
       'playFT%', 'playORB', 'playDRB', 'playTRB', 'opptAbbr', 'opptConf',
       'opptDiv', 'opptLoc', 'opptRslt', 'opptDayOff']]