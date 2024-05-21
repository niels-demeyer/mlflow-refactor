import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class RandomForestMLflow:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.clean_df()
        self.split_data()
        self.create_pipeline()

    def clean_df(self):
        self.df = self.df[self.df.columns[:-2]]
        self.df = self.df.drop(["CLIENTNUM"], axis=1)

    def split_data(self):
        X = self.df.drop("Attrition_Flag", axis=1)
        y = self.df["Attrition_Flag"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def create_pipeline(self):
        # Identifying categoricals and numericals
        categorical_cols = self.X_train.select_dtypes(
            include=["object", "category"]
        ).columns
        numerical_cols = self.X_train.select_dtypes(
            exclude=["object", "category"]
        ).columns

        # Numerical preprocessing
        numerical_pipeline = make_pipeline(
            SimpleImputer(strategy="mean"), StandardScaler()
        )

        # Categorical preprocessing
        categorical_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))

        # ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_pipeline, categorical_cols),
                ("num", numerical_pipeline, numerical_cols),
            ],
            remainder="passthrough",
        )
