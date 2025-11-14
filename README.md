# Multi-Factor-Regime-Adaptive-Exposure-Strategy-
A systematic exposure strategy that adapts to market regimes using multi-factor momentum, volatility targeting, and nonlinear signal processing for more stable and risk-aware asset allocation.
Multi-Factor Regime Adaptive Exposure Strategy

# Overview

This project implements a daily exposure strategy that integrates momentum factors, realized volatility, and market regime conditions. The model applies nonlinear transformations and volatility-aware weighting to produce smooth, risk-adjusted exposure values suitable for systematic trading pipelines.

# Methodology
	•	Long-term and short-term momentum factors
	•	Realized volatility with inverse-volatility weighting
	•	Regime filters based on volatility and short-horizon return conditions
	•	Nonlinear activation and smoothing to reduce noise
	•	Exposure clipping to maintain leverage constraints

# Backtesting:
	•	Strategy return
	•	CAGR
	•	Maximum drawdown
	•	Sharpe ratio
	•	Calmar ratio
	•	Benchmark comparison against buy-and-hold



# Data Requirements

This project does not include raw financial datasets due to licensing restrictions.

To run the Adaptive Exposure Model, you must provide a CSV file with the following columns:

- QQQ_close
- QQQ_today_close_to_tmrw_close_return
- VIX
- LT_Score
- ST_Score
- Or you can use the sample data provided under QQQ_base.xlsx

Save your file in this directory as:

    data/qqq_features.csv

The file must contain daily data with a Date column or index.
