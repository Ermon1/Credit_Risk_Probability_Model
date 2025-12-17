import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class CustomerFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, snapshot_date=None, selected_features=None):
        self.snapshot_date = snapshot_date
        self.selected_features = selected_features

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
        df["transaction_hour"] = df["TransactionStartTime"].dt.hour
        df["transaction_date"] = df["TransactionStartTime"].dt.date
        df["is_weekend"] = df["TransactionStartTime"].dt.weekday >= 5
        df["is_refund"] = df["Amount"] < 0

        snapshot_date = (
            pd.to_datetime(self.snapshot_date)
            if self.snapshot_date
            else df["TransactionStartTime"].max() + pd.Timedelta(days=1)
        )

        agg = df.groupby("CustomerId").agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            median_amount=("Amount", "median"),
            amount_std=("Amount", "std"),
            transaction_count=("TransactionId", "count"),
            active_days=("transaction_date", "nunique"),
            last_transaction=("TransactionStartTime", "max"),
            avg_transaction_hour=("transaction_hour", "mean"),
            std_transaction_hour=("transaction_hour", "std"),
            weekend_ratio=("is_weekend", "mean"),
            refund_ratio=("is_refund", "mean"),
            num_unique_product_categories=("ProductCategory", "nunique"),
        ).reset_index()

        agg["recency_days"] = (snapshot_date - agg["last_transaction"]).dt.days
        agg["log_total_amount"] = np.log1p(np.abs(agg["total_amount"]))

        category_counts = df.groupby(["CustomerId", "ProductCategory"]).size().unstack(fill_value=0)
        category_ratios = category_counts.div(category_counts.sum(axis=1), axis=0).reset_index()

        for col in ["financial_services", "airtime"]:
            if col not in category_ratios.columns:
                category_ratios[col] = 0.0

        category_ratios = category_ratios[["CustomerId", "financial_services", "airtime"]].rename(
            columns={"financial_services": "pct_financial_services", "airtime": "pct_airtime"}
        )

        features = agg.merge(category_ratios, on="CustomerId", how="left")
        features = features.drop(columns=["last_transaction"]).fillna(0)

        if self.selected_features:
            features = features[["CustomerId"] + self.selected_features]

        return features


def build_feature_pipeline(snapshot_date=None, selected_features=None) -> Pipeline:
    return Pipeline(
        steps=[
            ("customer_feature_engineering", CustomerFeatureEngineer(snapshot_date, selected_features))
        ]
    ) 
         

def process_and_save_data(config):
    """
    Load raw data, transform, and save processed features
    """
    # Load raw
    df_raw = pd.read_csv(config["data"]["source"])
    
    # Build pipeline
    pipeline = build_feature_pipeline(
        snapshot_date=config["snapshot_date"],
        selected_features=config["features"]["include"]
    )
    
    # Transform
    df_features = pipeline.fit_transform(df_raw)
    
    # Save processed
    df_features.to_csv(config["data"]["processed"], index=False)
    print(f"Processed features saved to {config['data']['processed']}")
    return df_features      

