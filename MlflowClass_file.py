import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


class RandomForestMLflow:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.clean_df()
        self.split_data()

    def clean_df(self):
        self.df = self.df[self.df.columns[:-2]]
        self.df = self.df.drop(["CLIENTNUM"], axis=1)

    def split_data(self):
        X = self.df.drop("Attrition_Flag", axis=1)
        y = self.df["Attrition_Flag"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
