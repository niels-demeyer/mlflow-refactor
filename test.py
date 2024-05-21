from MlflowClass_file import RandomForestMLflow

ml = RandomForestMLflow()


# define the filepath
filepath = "./customer_churn_analysis/data/BankChurners.csv"
rf = RandomForestMLflow(filepath=filepath)
rf.train_model()
rf.evaluate_model()
rf.mlflow_run()
