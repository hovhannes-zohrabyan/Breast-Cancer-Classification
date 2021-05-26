import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetController:

    X_train, X_test, y_train, y_test = None, None, None, None

    def __init__(self):
        try:
            self.df = pd.read_csv('../Data/breast_cancer.csv')
        except FileNotFoundError:
            exit("Dataset not found, exiting the program")

        self.data_cleaning()
        self.train_test_split()

    def data_cleaning(self):
        df = self.df.drop(['id', 'Unnamed: 32'], axis=1)
        # TODO: Add data cleaning, outliers
        self.df = df

    def train_test_split(self):
        y = self.df['diagnosis']
        X = self.df.drop('diagnosis', axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
    # Test changes