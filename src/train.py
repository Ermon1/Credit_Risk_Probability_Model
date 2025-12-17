import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from src.modeling.models import get_models
from src.modeling.evaluation import evaluate_model
from src.utility.config_loader import config


def main():
    data_path = config["data"]["processed"]
    df = pd.read_csv(data_path)

    TARGET = "is_high_risk"

    X = df.drop(columns=[TARGET, "CustomerId"])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    mlflow.set_experiment("credit-risk-model")

    models = get_models()

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="CreditRiskModel",
            )


if __name__ == "__main__":
    main()
