from cmath import nan
from locale import D_FMT
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost as XGB

from xgboost import XGBRegressor
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

def find_bad_columns(df):
    """ Inspect data for missing values etc """
    null_columns = []
    for col in df.columns:
        missing_values = df[col].isnull()
        if missing_values.sum() > 100:
            print(f"{col} has {missing_values.sum()} missing values")
 

def preprocess(df ):
    categorical_cols = ["LotConfig","LotArea","LandSlope","Neighborhood","Condition1",\
                       "Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl",\
                       "Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond",\
                       "Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
                       "BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","YearRemodAdd",\
                       "TotalBsmtSF","Heating","HeatingQC","CentralAir","Utilities",\
                       "Electrical","KitchenQual","Functional","PavedDrive","SaleType",\
                       "GarageType","GarageFinish","GarageQual","GarageCond",\
                       "SaleCondition","MSZoning","LotShape","Street","LandContour"]

    df['MasVnrType'].fillna('None', inplace=True)
    df['BsmtQual'].fillna('None', inplace=True)
    df['BsmtCond'].fillna('None', inplace=True)
    df['BsmtExposure'].fillna('None', inplace=True)
    df['BsmtFinType1'].fillna('None', inplace=True)
    df['BsmtFinType2'].fillna('None', inplace=True)
    df['Electrical'].fillna('None', inplace=True)
    df['GarageType'].fillna('None', inplace=True)
    df['GarageFinish'].fillna('None', inplace=True)
    df['GarageQual'].fillna('None', inplace=True)
    df['GarageCond'].fillna('None', inplace=True)
    df['Exterior1st'].fillna('None', inplace=True)
    df['Exterior2nd'].fillna('None', inplace=True)
    df['Utilities'].fillna('None', inplace=True)
    df['Electrical'].fillna('None', inplace=True)
    df['KitchenQual'].fillna('None', inplace=True)
    df['Functional'].fillna('None', inplace=True)
    df['SaleType'].fillna('None', inplace=True)
    df['MSZoning'].fillna('None', inplace=True)

    """ Mean value of lotarea is too large, needs to be harmonized wiht other features"""
    df['LotArea'] = np.log1p(df['LotArea'])

    df['LotFrontage'].fillna(np.mean(df['LotFrontage']), inplace=True)
    df['MasVnrArea'].fillna(0, inplace=True)
    df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
    

    df['MasVnrArea'] = df['MasVnrArea'].astype(int)
    return df

def remove_skew(df):
    numeric_feats = df.dtypes[df.dtypes != 'object'].index
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.5]
    
    for feature in high_skew.index:
        df[feature] = np.log1p(df[feature])
    return df


def rmse_CV_train(model):
    kf = KFold(5,shuffle=True,random_state=42).get_n_splits(x_train.values)
    rmse = np.sqrt(-cross_val_score(model, x_train, y_train,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)
def rmse_CV_test(model):
    kf = KFold(5,shuffle=True,random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, x_test, y_test,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)

def main():
    train_data_location = "data/train.csv"
    test_data_location = "data/test.csv"

    train_df = pd.read_csv(train_data_location)
    test_df = pd.read_csv(test_data_location)
    
    print(train_df.shape)
    #print(train_df.info)
    #print("Skewness: %f" % train_df['SalePrice'].skew())
    #print("Kurtosis: %f" % train_df['SalePrice'].kurt())
    train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

    # Look at average year built and sale price in each zone
    grouped = train_df['YearBuilt'].groupby(train_df['MSZoning'])
    #print(grouped.mean())

    grouped = train_df['SalePrice'].groupby(train_df['MSZoning'])
    #print(grouped.mean())

    # Lets look at the skew of the saleprice so we are aware of bias...
    #ax = sns.histplot(train_df['SalePrice'])
    #plt.show()
    
    find_bad_columns(train_df)
    null_cols = ["Alley","PoolQC","Fence","MiscFeature","FireplaceQu"]

    """ Drop Features with lots of nulls"""
    train_df.drop(columns=null_cols, inplace=True)
    test_df.drop(columns=null_cols, inplace=True)
    
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    """ Remove duplicates """
    #print("former shape: ", train_df.shape, " test : ", test_df.shape)
    train_df.drop_duplicates()
    test_df.drop_duplicates()
    #print("latter shape: ", train_df.shape, " test : ", test_df.shape)

    """ Check for important features -> use correlation matrix """
    corrmat = train_df.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True);
    
    # View Important Features
    corr = train_df.corr()
    highest_corr_features = corr.index[(corr["SalePrice"])>0.5]
    plt.figure(figsize=(10,10))
    g = sns.heatmap(train_df[highest_corr_features].corr(),annot=True,cmap="RdYlGn")
    #print(corr["SalePrice"].sort_values(ascending=False))

    # Features with highest values 
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(train_df[cols])
    #plt.show()

    # Overall Quality is highest correlation, garagecars and area are the same...
    # Total BsmtSF and 1stFlrSF are like each other 
    # Total Rms abv grd and grliv area have strong correlation 

    # Fix Skew in data...
    train_df = remove_skew(train_df)
    test_df = remove_skew(test_df)

    train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
    test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']

    scorer = make_scorer(mean_squared_error,greater_is_better = False)

    the_model = XGB.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                                learning_rate=0.05, max_depth=3, 
                                min_child_weight=1.7817, n_estimators=2200,
                                reg_alpha=0.4640, reg_lambda=0.8571,
                                subsample=0.5213, random_state =7, nthread = -1)
    
    y_train = train_df['SalePrice']
    train_df = train_df.drop(['Id', 'SalePrice'], axis=1)
    sub = pd.DataFrame()
    sub['Id'] = test_df['Id']
    test_df = test_df.drop(['Id'], axis=1)

    the_model.fit(train_df, y_train)

    y_predict = np.floor(np.expm1(the_model.predict(test_df)))
    print(y_predict)
    sub['SalePrice'] = y_predict
    sub.to_csv('mysubmission.csv',index=False)

if __name__ == "__main__":
    main()