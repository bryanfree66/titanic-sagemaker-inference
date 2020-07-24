def get_title(name):
    title_search = re.search(r'([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

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

def feature_eng(X):
    # Rename Categorical Columns
    
    # Create title feature
    X['Title'] = X['Name'].apply(get_title)
    X['Title'] = X['Title'].fillna('Miss')
    X['Title'] = X['Title'].apply(replace_titles)
    
    # Drop Name
    X.drop('Name', axis=1, inplace=True)
    
    #Impute Age
    X.loc[X.Age.isnull(), 'Age'] = X.groupby(['Sex','Pclass','Title']).Age.transform('median')
    
    # Convert Pclass
    X['Pclass'] = X['Pclass'].apply(lambda x: 'first' if x==1 else 'second' if x==2 else 'third')
    
    # Create Age Bins
    binner = KBinsDiscretizer(encode='ordinal')
    binner.fit(X[['Age']])
    X['AgeBins'] = binner.transform(X[['Age']])
    
    # Create family size feature
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    
    # Family size mapping
    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 
                  5: 'Large', 6: 'Large', 7: 'Large', 8: 'Large', 11: 'Large'}
    X['GroupSize'] = X['FamilySize'].map(family_map)
    
    # With Family Feature
    X['WithFamily'] = (X['FamilySize']>1)
    X['WithFamily'] = X['WithFamily'].apply(lambda x: 'yes' if x==1 else 'no')
    
    # Impute Fares
    X.loc[(X.Fare.isnull()), 'Fare'] = X.Fare.median()
    X.loc[(X.Fare==0), 'Fare'] = X.Fare.median()
    
    # Create Fare Bins
    binner.fit(X[['Fare']])
    X['FareBins'] = binner.transform(X[['Fare']])
    
    # Create deck and room features
    X["Deck"] = X["Cabin"].str.slice(0,1)
    X["Deck"] = X["Deck"].fillna("N")
    idx = X[X['Deck'] == 'T'].index
    X.loc[idx, 'Deck'] = 'A'
    
    X["Room"] = X["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False)
    X['Room'] = X['Room'].astype(str)
    X["Room"] = X["Room"].fillna('None')
    
    # Drop Cabin
    X.drop('Cabin', axis=1, inplace=True)
    
    # Impute Embarked
    encoder=OrdinalEncoder()
    X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)

    # Add some log features
    X['LogAge'] = np.log(X['Age'])
    X['LogFare'] = np.log(X['Fare'])
    X['LogFamilySize'] = np.log(X['Fare'])
    
    # Add scaled features
    scaler = StandardScaler()
    scaler.fit(X[['Age']])
    X['AgeScaled'] = scaler.transform(X[['Age']])
    
    scaler.fit(X[['Fare']])
    X['FareScaled'] = scaler.transform(X[['Fare']])
    
    scaler.fit(X[['FamilySize']])
    X['FareScaled'] = scaler.transform(X[['Fare']])
    
    # Get Dummies
    X = pd.get_dummies(X, prefix='dummy', drop_first=True)
    X.drop('PassengerId', axis=1, inplace=True)
    X.drop('Age', axis=1, inplace=True)
    X.drop('Parch', axis=1, inplace=True)
    X.drop('SibSp', axis=1, inplace=True)
    X.drop('Fare', axis=1, inplace=True)
    return X