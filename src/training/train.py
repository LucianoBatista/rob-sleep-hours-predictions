import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pmdarima as pm
import polars as pl
import yaml
from pmdarima.model_selection import train_test_split
from sklearn import metrics
from statsmodels.tsa.stattools import adfuller


class Trainer:
    def __init__(
        self,
        train_path: str | Path,
        test_path: str | Path,
        metrics_path: str | Path,
        arima_model_path: str | Path,
        sub_name: str,
    ) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.metrics_path = metrics_path
        self.arima_model_path = arima_model_path
        self.sub_name = sub_name

    def train_baseline(self) -> None:
        # train data
        train_data = pl.read_parquet(self.train_path)

        # imputing missing values
        train_data_interpolated = train_data.interpolate()
        train, test = train_test_split(
            train_data_interpolated["sleep_hours"], train_size=0.9
        )

        ARIMA_model = pm.auto_arima(
            train,
            start_p=1,
            start_q=1,
            start_P=1,
            start_Q=1,
            max_p=5,
            max_q=5,
            max_P=5,
            max_Q=5,
            seasonal=True,
            stepwise=True,
            suppress_warnings=True,
            D=10,
            max_D=10,
            error_action="ignore",
        )

        # Create predictions for the future, evaluate on test
        preds, conf_int = ARIMA_model.predict(
            n_periods=test.shape[0], return_conf_int=True
        )

        # saving the model
        with open(self.arima_model_path, "wb") as pkl:
            pickle.dump(ARIMA_model, pkl)

        # saving the metrics
        with open(self.metrics_path, "w") as f:
            # saving json file with RMSE
            rmse = np.sqrt(metrics.mean_squared_error(test, preds))
            json.dump({"rmse": float(rmse)}, f)

    def create_submission(self) -> None:
        # expected periods to predict
        periods_to_predict = pl.date_range(
            datetime(2022, 1, 1), datetime(2023, 3, 16), "1d", name="date"
        )

        # loading trained model
        with open(self.arima_model_path, "rb") as pkl:
            pickle_preds, conf_int = pickle.load(pkl).predict(
                n_periods=periods_to_predict.shape[0], return_conf_int=True
            )

        submission_df = pl.DataFrame(
            {"date": periods_to_predict, "sleep_hours": pickle_preds}
        )
        test_df = pl.read_csv(self.test_path)

        test_dt_df = test_df.with_columns(
            [pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d").alias("date")]
        )
        test_df_sub = test_dt_df.join(
            submission_df,
            on="date",
            how="left",
        )

        submission_final = test_df_sub.drop("sleep_hours").rename(
            {"sleep_hours_right": "sleep_hours"}
        )
        submission_final.with_columns([pl.col("date").cast(pl.Date)]).write_csv(
            "data/pog-sleep-data/subs/{self.sub_name}.csv"
        )


if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_path = config["train"]["train_adjusted_path"]
    test_path = config["data"]["test_path"]
    metrics_path = config["train"]["metrics_path"]
    arima_model_path = config["train"]["arima_model_path"]
    sub_name = config["train"]["sub_name"]

    trainer = Trainer(
        train_path=train_path,
        test_path=test_path,
        metrics_path=metrics_path,
        arima_model_path=arima_model_path,
        sub_name=sub_name,
    )
    trainer.train_baseline()
    trainer.create_submission()
