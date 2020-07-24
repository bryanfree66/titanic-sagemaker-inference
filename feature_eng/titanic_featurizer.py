import time
import sys
from io import StringIO
import os
import shutil
import pdb
import re

import argparse
import csv
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder, KBinsDiscretizer, OrdinalEncoder

'''
Description: Helper function to derive passenger title from name with regex
Parameter: name - string
Returns: string (unformatted title)
'''
def get_title(name):
    title_search = re.search(r'([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

'''
Description: Helper function group and replace the unformatted titles returned by get-title
Parameter: title - string
Returns: string
'''
def replace_titles(title):
    if title in ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
        return 'Rare'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

'''
Description: Main feaure engineering method
Parameter: Pandas data frame
Return: Pandas data frame
'''
def feature_eng(X):
    X['Title'] = X['Name'].apply(get_title)
    X['Title'] = X['Title'].fillna('Miss')
    X['Title'] = X['Title'].apply(replace_titles)
    X.loc[X.Age.isnull(), 'Age'] = X.groupby(['Sex','Pclass','Title']).Age.transform('median')
    X['Pclass'] = X['Pclass'].apply(lambda x: 'first' if x==1 else 'second' if x==2 else 'third')
    binner = KBinsDiscretizer(encode='ordinal')
    binner.fit(X[['Age']])
    X['AgeBins'] = binner.transform(X[['Age']])
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 
                  5: 'Large', 6: 'Large', 7: 'Large', 8: 'Large', 11: 'Large'}
    X['GroupSize'] = X['FamilySize'].map(family_map)
    X['WithFamily'] = (X['FamilySize']>1)
    X['WithFamily'] = X['WithFamily'].apply(lambda x: 'yes' if x==1 else 'no')
    X.loc[(X.Fare.isnull()), 'Fare'] = X.Fare.median()
    X.loc[(X.Fare==0), 'Fare'] = X.Fare.median()
    binner.fit(X[['Fare']])
    X['FareBins'] = binner.transform(X[['Fare']])
    X["Deck"] = X["Cabin"].str.slice(0,1)
    X["Deck"] = X["Deck"].fillna("N")
    idx = X[X['Deck'] == 'T'].index
    X.loc[idx, 'Deck'] = 'A'
    X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)
    X.drop('PassengerId', axis=1, inplace=True)
    X.drop('Ticket', axis=1, inplace=True)
    X.drop('Name', axis=1, inplace=True)
    return X

# Define the feature columns, since we get a headless CSV
input_feature_columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 
                         'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    
target_col = 'Survived'

input_feature_dtypes = {
    'PassengerId': np.int64,
    'Pclass': np.int64,
    'Sex': str,
    'Age': np.float64,
    'SibSp': np.int64,
    'Parch': np.int64,
    'Ticket': str,
    'Fare': np.float64,
    'Cabin': str,
    'Embarked': str
    }

target_dtype = {'Survived': np.int64}

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

'''
Description: main method
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))

    raw_data = [ pd.read_csv(
        file, 
        header=None, 
        names=input_feature_columns + [target_col],
        dtype=merge_two_dicts(input_feature_dtypes, target_dtype), engine='python') for file in input_files ]
    concat_data = pd.concat(raw_data)

    # Create new features
    data_eng = feature_eng(concat_data)

    eng_feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                           'Title', 'AgeBins', 'FamilySize', 'GroupSize', 'WithFamily',
                           'FareBins', 'Deck']

    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
   
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
        ])

    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'GroupSize']
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder="drop")

    preprocessor.fit(concat_data)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))