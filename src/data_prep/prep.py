import calendar
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml


def get_max_days_by_month(year, month=1):
    month_days = calendar.monthcalendar(year, month)
    return max(month_days[-1])


def prep_data(train_path: str | Path):
    # training data transformation
    train_data = pl.read_csv(train_path)
    train_data_dt = train_data.with_columns(
        [pl.col("date").str.strptime(pl.Datetime, fmt="%Y-%m-%d")]
    )

    train_data_dt_ex_vars = train_data_dt.with_columns(
        [
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.day().alias("day"),
        ]
    )

    train_data_grouped_year_month = (
        train_data_dt_ex_vars.groupby(["year", "month"]).count().sort(["year", "month"])
    )

    # how many days in each month
    max_days = []
    for rows in train_data_grouped_year_month.rows():
        year = rows[0]
        month = rows[1]
        max_days.append(get_max_days_by_month(year, month))

    # the expected range of dates
    correct_date_range = pl.date_range(
        datetime(2015, 2, 19), datetime(2021, 12, 31), "1d", name="date"
    )
    correct_date_range_df = pl.DataFrame({"date": correct_date_range})

    # replacing the original date column with the expected range of dates
    train_data_fixed_df = correct_date_range_df.join(
        train_data_dt,
        on="date",
        how="left",
    )

    train_data_fixed_df.write_parquet("data/pog-sleep-data/train_dt_fixed.parquet")


if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_path = config["data"]["train_path"]
    train_data = prep_data(train_path)
