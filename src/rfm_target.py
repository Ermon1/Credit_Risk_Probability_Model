import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class RFMTargetEngineer:
    """
    Task 4 ONLY:
    Create proxy credit risk target using RFM clustering
    """

    def __init__(self, n_clusters=3, random_state=42):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=50
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        rfm = df[["recency_days", "transaction_count", "total_amount"]]
        rfm_scaled = self.scaler.fit_transform(rfm)
        df["cluster"] = self.kmeans.fit_predict(rfm_scaled)

        profile = df.groupby("cluster").agg(
            recency=("recency_days", "mean"),
            frequency=("transaction_count", "mean"),
            monetary=("total_amount", "mean"),
        )

        high_risk_cluster = profile.sort_values(
            by=["recency", "frequency", "monetary"],
            ascending=[False, True, True]
        ).index[0]

        df["is_high_risk"] = (df["cluster"] == high_risk_cluster).astype(int)
        df = df.drop(columns=["cluster"])

        return df
