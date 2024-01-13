import pandas as pd 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler

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

rr = RidgeClassifier(alpha = 1) #machine learning model
split = TimeSeriesSplit(n_splits=3) 
#feature is just a col
#train model using sets of features uses sets of features to see which ones work best
#Ex. if the input data(features) lead to an accurate prediction of outcome of a game using this feature is good
#direction means it will start with no feature and add the one that improve model accuracy the most
#cv = split to make sure we are not using same data in training and prediction
sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction='forward', cv = split)  

removed_cols = ['season', 'date', 'won', 'target', 'team', 'team_opp'] #list of cols we dont want to scale

selected_cols = data.columns[~data.columns.isin(removed_cols)]
scalar = MinMaxScaler()
data[selected_cols] = scalar.fit_transform(data[selected_cols]) #makes all cols not in removed_cols between 0 and 1

sfs.fit(data[selected_cols], data['target'])    #fit based on selected cols to predict target(pick 30 best features to predict target)

predictors = list(selected_cols[sfs.get_support()]) #list of 30 cols used in predictor

def backtest (data, model, predictors, start=2, step=1):#split data by season to predict future seasons using past data
    #start = 2 so we use at least 2 seasons to predict next season 
    all_predictions = []
    seasons = sorted(data['season'].unique())#list of unique seasons
    for i in range(start, len(seasons),step):#from start to however many unique seasons
        season = seasons[i] 

        train = data[data['season'] < season]#data we train with(all prev seasons)
        test = data[data['season'] == season]#data we are predicting(current season)

        model.fit(train[predictors], train['target']) #fit data based on predictors to predict target
        preds = model.predict(test[predictors]) #predict on test based on training
        preds = pd.Series(preds, index = test.index) #convert numpy arr to pd seris

        combined = pd.concat([test['target'], preds], axis = 1) #2 cols one with actual data and one with predicted
        combined.columns = ['actual', 'prediction'] #rename cols

        all_predictions.append(combined) #axis 0
    return pd.concat(all_predictions)
predictions = backtest(data, rr, predictors)
print(predictions)