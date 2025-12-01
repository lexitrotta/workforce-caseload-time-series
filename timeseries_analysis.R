# r/timeseries_analysis.R

library(tidyverse)
library(forecast)

data_path <- "data/processed/caseload_monthly.csv"
df <- read_csv(data_path)

# Assume year_month is YYYY-MM-01
df_ts <- ts(df$total_cases, frequency = 12)

autoplot(df_ts) + ggtitle("Monthly Case Load")

# Decompose
fit <- stl(df_ts, s.window = "periodic")
autoplot(fit)

# Simple forecast example
fc <- forecast(fit, h = 12)
autoplot(fc)
