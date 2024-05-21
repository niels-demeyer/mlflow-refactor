import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class RandomForestMLflow:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, target, test_size=0.2, random_state=42
        )

    def train_model(self, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        self.predictions = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.predictions)

    def log_mlflow(self):
        with mlflow.start_run():
            mlflow.log_param("n_estimators", self.model.n_estimators)
            mlflow.log_param("max_depth", self.model.max_depth)
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.sklearn.log_model(self.model, "model")
