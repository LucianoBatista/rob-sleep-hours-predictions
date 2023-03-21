import itertools
import subprocess

if __name__ == "__main__":
    # Automated grid search experiments
    arima_max_iter = [100, 200, 300]
    arima_n_fits = [10, 50, 100]
    arima_m = [1, 12, 52]

    # Iterate over all combinations of hyperparameter values.
    for x_arima_m, x_arima_n_fits, x_arima_max_iter in itertools.product(
        arima_m, arima_n_fits, arima_max_iter
    ):
        subprocess.run(
            [
                "dvc",
                "exp",
                "run",
                "-f",
                "--queue",
                "--set-param",
                f"train.arima_n_fits={x_arima_n_fits}",
                "--set-param",
                f"train.arima_maxiter={x_arima_max_iter}",
                "--set-param",
                f"train.arima_m={x_arima_m}",
            ]
        )
