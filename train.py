import pandas as pd




def main():

   train_data_location = "data/train.csv"
   df = pd.read_csv(train_data_location)

   print(list(df.columns))


if __name__ == "__main__":
    main()