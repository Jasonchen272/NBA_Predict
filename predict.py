import pandas as pd 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier

data = pd.read_csv('nba_games.csv', index_col=0)    #read from csv

data = data.sort_values('date')     #sort by date
data = data.reset_index(drop=True)  #reset index to correlate with date

del data['mp.1']    #delete excess cols
del data['mp_opp.1']
del data['index_opp']

def add_target(team):   #adds target col that is if argument team won their next game
    team['target'] = team['won'].shift(-1)
    return team

data = data.groupby('team', group_keys=False).apply(add_target) #add_target on each team independantly 
data['target'][pd.isnull(data['target'])] = 2   #no next game means target is null so set it to 2
data['target']   = data['target'].astype(int, errors = 'ignore')    #target = True --> 1 target = False --> 0

nulls = pd.isnull(data) 
nulls = nulls.sum()
nulls = nulls[nulls>0]  #all cols with null vals

valid_cols = data.columns[~data.columns.isin(nulls.index)]  #remove null vals from data

data = data[valid_cols].copy()  #copy non-null cols to data

