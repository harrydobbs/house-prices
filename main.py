import pandas as pd

from preprocessing import *


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
   preprocess_data(train_data_location, test_data_location)

   print(list(df.columns))


if __name__ == "__main__":
    main()