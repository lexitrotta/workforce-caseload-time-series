# scripts/timeseries_arima.py

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

DATA_PATH = Path("data/processed/caseload_monthly.csv")


def load_ts():
    df = pd.read_csv(DATA_PATH, parse_dates=["year_month"])
    df = df.sort_values("year_month")
    df = df.set_index("year_month").asfreq("MS")  # Monthly Start
    ts = df["total_cases"]
    return ts


def adf_test(series, name="time series"):
    """
    Quick wrapper to print ADF test results.
    """
    result = adfuller(series.dropna())
    print(f"ADF test for {name}")
    print(f"  Test statistic: {result[0]:.4f}")
    print(f"  p-value:        {result[1]:.4f}")
    for key, value in result[4].items():
        print(f"  Critical value ({key}): {value:.4f}")
    print("-" * 40)


def main():
    ts = load_ts()

    # 1. Plot raw series
    plt.figure()
    ts.plot()
    plt.title("Monthly Case Load")
    plt.xlabel("Date")
    plt.ylabel("Total Cases")
    plt.tight_layout()
    plt.show()

    # 2. Stationarity check
    adf_test(ts, name="total_cases")

    # If non-stationary, difference once (d=1)
    ts_diff = ts.diff().dropna()

    # 3. ACF/PACF on differenced series (to choose p, q)
    plt.figure()
    plot_acf(ts_diff, lags=24)
    plt.title("ACF of Differenced Series")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plot_pacf(ts_diff, lags=24)
    plt.title("PACF of Differenced Series")
    plt.tight_layout()
    plt.show()

    # 4. Fit ARIMA model
    # Example: ARIMA(1,1,1) – you can tune later based on AIC / diagnostics
    model = ARIMA(ts, order=(1, 1, 1))
    model_fit = model.fit()
    print(model_fit.summary())

    # 5. In-sample fitted values vs actual
    plt.figure()
    ts.plot(label="Actual")
    model_fit.fittedvalues.plot(label="Fitted", linestyle="--")
    plt.title("ARIMA(1,1,1) – Actual vs Fitted")
    plt.xlabel("Date")
    plt.ylabel("Total Cases")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6. Forecast next 12 months
    n_forecast = 12
    forecast_res = model_fit.get_forecast(steps=n_forecast)
    forecast_mean = forecast_res.predicted_mean
    forecast_ci = forecast_res.conf_int()

    plt.figure()
    ts.plot(label="History")
    forecast_mean.plot(label="Forecast", linestyle="--")

    # Confidence interval shading
    plt.fill_between(
        forecast_ci.index,
        forecast_ci["lower total_cases"],
        forecast_ci["upper total_cases"],
        alpha=0.2,
    )

    plt.title(f"ARIMA(1,1,1) Forecast – Next {n_forecast} Months")
    plt.xlabel("Date")
    plt.ylabel("Total Cases")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
