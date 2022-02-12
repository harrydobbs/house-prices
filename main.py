import pandas as pd

from preprocessing import *


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def main():

   train_data_location = "data/train.csv"
   test_data_location = "data/test.csv"
   train_df, test_df = preprocess_data(train_data_location, test_data_location)

   xgb_param = {'learning_rate': 0.03,
                'max_depth': 40,
                'verbosity': 3,
                'nthread': 5,
                'random_state': 0,
                'subsample': 0.7,
                'n_estimators': 5000,
                'colsample_bytree': 0.8}


   model_xgb = XGBRegressor(learning_rate=xgb_param['learning_rate'],
                            max_depth=xgb_param['max_depth'],
                            verbosity=xgb_param['verbosity'],
                            nthread=xgb_param['nthread'],
                            random_state=xgb_param['random_state'],
                            subsample=xgb_param['subsample'],
                            n_estimators=xgb_param['n_estimators'],
                            colsample_bytree=xgb_param['colsample_bytree'])

    
   y_train = train_df['SalePrice']
   train_df = train_df.drop(['Id', 'SalePrice'], axis=1)
   sub = pd.DataFrame()
   sub['Id'] = test_df['Id']
   test_df = test_df.drop(['Id'], axis=1)

   model_xgb.fit(train_df, y_train)
   xgb_train_pred = model_xgb.predict(train_df)
   print(rmsle(y_train, xgb_train_pred))

   y_predict = np.floor(np.expm1(model_xgb.predict(test_df)))
   print(y_predict)
   sub['SalePrice'] = y_predict
   sub.to_csv('mysubmission.csv',index=False)

if __name__ == "__main__":
    main()