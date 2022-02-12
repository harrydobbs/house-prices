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


def clean_df(df):
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

    """ Mean value of lotarea is too large, needs to be harmonized with other features"""
    df['LotArea'] = np.log1p(df['LotArea'])

    df['LotFrontage'].fillna(np.mean(df['LotFrontage']), inplace=True)
    df['MasVnrArea'].fillna(0, inplace=True)
    df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
    
    df['MasVnrArea'] = df['MasVnrArea'].astype(int)

    df.drop_duplicates()
    return df


def remove_skew(df):
    numeric_feats = df.dtypes[df.dtypes != 'object'].index
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.5]
    for feature in high_skew.index:
        df[feature] = np.log1p(df[feature])
    return df


def eda(df):
    print(df.shape)
    print(df.info)
    print("Skewness: %f" % df['SalePrice'].skew())
    print("Kurtosis: %f" % df['SalePrice'].kurt())
    
    # Look at average year built and sale price in each zone
    grouped = df['YearBuilt'].groupby(df['MSZoning'])
    print(grouped.mean())
    grouped = df['SalePrice'].groupby(df['MSZoning'])
    print(grouped.mean())

    # Lets look at the skew of the saleprice so we are aware of bias...
    ax = sns.histplot(df['SalePrice'])

    """ Check for important features -> use correlation matrix """
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True);
    
    corr = df.corr()
    highest_corr_features = corr.index[(corr["SalePrice"])>0.5]
    plt.figure(figsize=(10,10))
    g = sns.heatmap(df[highest_corr_features].corr(),annot=True,cmap="RdYlGn")
    g.set_title('Correlation Heatmap', fontdict={'fontsize':8}, pad=12)
    print(corr["SalePrice"].sort_values(ascending=False))

    # Features with highest values 
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(df[cols])
    plt.show()


def add_features(df):
    df['house_age1'] = df['YrSold'] - df['YearBuilt']
    df['house_age2'] = df['YrSold'] - df['YearRemodAdd']
    df['garage_age'] = df['YrSold'] - df['GarageYrBlt']
    
    df['total_area'] = np.log1p(df['GrLivArea'] + df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'])
    df['num_rooms'] = df['TotRmsAbvGrd'] + df['BedroomAbvGr'] + df['FullBath']
    return df



def preprocess_data(train_data_location, test_data_location):

    # Load Data Frames
    train_df = pd.read_csv(train_data_location)
    test_df = pd.read_csv(test_data_location)
    
    # It is observed that Sale Price is right skewed, so we do a log 
    # transformation
    train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

    """ Drop Features with lots of nulls"""
    print(train_df.isnull().sum())
    null_cols = ["Alley","PoolQC","Fence","MiscFeature","FireplaceQu"]
    train_df.drop(columns=null_cols, inplace=True)
    test_df.drop(columns=null_cols, inplace=True)
    
    train_df = clean_df(train_df)
    test_df = clean_df(test_df)
    train_df = remove_skew(train_df)
    test_df = remove_skew(test_df)

    # Inspect data 
    eda(train_df)
    add_features(train_df)
    add_features(test_df)

    train_df['YrSold'] = train_df['YrSold'].replace({2008:2, 
                                                 2007:1, 
                                                 2006:0, 
                                                 2009:3, 
                                                 2010:4})
    test_df['YrSold'] = test_df['YrSold'].replace({2008:2, 
                                                 2007:1, 
                                                 2006:0, 
                                                 2009:3, 
                                                 2010:4})
    return train_df, test_df



if __name__ == "__main__":
   train_data_location = "data/train.csv"
   test_data_location = "data/test.csv"
   preprocess_data(train_data_location, test_data_location)