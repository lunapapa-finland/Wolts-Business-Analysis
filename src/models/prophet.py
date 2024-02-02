
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from itertools import product
import matplotlib.pyplot as plt
import os

def grid_search_prophet(train, test, holidays_df, param_grid):
    """
    Perform grid search for Prophet model tuning.

    Parameters:
    - train: DataFrame, training data
    - test: DataFrame, test data
    - holidays_df: DataFrame, holidays information
    - param_grid: dict, parameter grid for grid search

    Returns:
    - best_model: Prophet model with the best parameters
    - best_params: Best parameter combination
    - best_mse: Mean Squared Error corresponding to the best parameters
    """
    best_model = None
    best_params = None
    best_mse = float('inf')

    # Generate all combinations of parameters
    param_combinations = list(product(*param_grid.values()))

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))

        # Create and fit Prophet model
        model_prophet = Prophet(**param_dict)
        model_prophet.add_country_holidays(country_name='FI')
        model_prophet.fit(train[['DATE', 'ORDER_COUNT']].rename(columns={'DATE': 'ds', 'ORDER_COUNT': 'y'}))

        # Forecast using Prophet
        future_prophet = model_prophet.make_future_dataframe(periods=len(test))
        forecast_prophet = model_prophet.predict(future_prophet)
        predicted_prophet = forecast_prophet[['ds', 'yhat']].tail(len(test))

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(test['ORDER_COUNT'], predicted_prophet['yhat'])

        # Update best model and parameters if current MSE is lower
        if mse < best_mse:
            best_mse = mse
            best_model = model_prophet
            best_params = param_dict

    return best_model, best_params, best_mse

def fit_forecast_plot_evaluate_prophet(logger, train, test, data, holidays_df, country_name , result_path):
    """
    Fits Prophet model, forecasts, plots results, and returns Mean Squared Error (MSE).

    Parameters:
    - train (pd.DataFrame): The training data.
    - test (pd.DataFrame): The testing data.
    - data (pd.DataFrame): The complete dataset.
    - holidays_df (pd.DataFrame): DataFrame with holiday information.
    - country_name (str, optional): The name of the country for country-specific holidays. Default is 'FI'.

    Returns:
    - model_prophet (Prophet): The fitted Prophet model.
    - mse_prophet (float): Mean Squared Error of the Prophet model on the test data.
    """
    logger.info('====================PROPHET==================')
    param_grid = {
    'changepoint_prior_scale': [0.01, 0.05, 0.1],
    'seasonality_prior_scale': [5.0, 10.0, 15.0],
    'uncertainty_samples': [1000, 5000, 10000],
    'yearly_seasonality': [False],
    'weekly_seasonality': [True, False],
    'daily_seasonality': [True, False],
    'seasonality_mode': ['additive', 'multiplicative']
    }

    model_prophet, params_prophet, mse_prophet_train = grid_search_prophet(train, test, holidays_df, param_grid)
    logger.info(f'Best Params for prophet are {params_prophet}')
    logger.info(f"Mean Squared Error (Prophet): {mse_prophet_train}")

    # Forecast using Prophet
    future_prophet = model_prophet.make_future_dataframe(periods=len(test))
    forecast_prophet = model_prophet.predict(future_prophet)
    predicted_prophet = forecast_prophet[['ds', 'yhat']].tail(len(test))

    # Plot Prophet results
    plt.figure(figsize=(8, 5))
    plt.plot(data['DATE'], data['ORDER_COUNT'], label='Actual')
    plt.plot(test['DATE'], predicted_prophet['yhat'], label='Prophet Forecast', linestyle='dashed')
    plt.title('Prophet Fit')
    plt.legend()
    plt.xticks(rotation=90)  # Rotate x-axis labels

    # Create the target folder if it doesn't exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Save the image to the specified path
    plt.tight_layout()
    plt.savefig(f"{result_path}/Prophet_Fit.png")


def order_forecast_prophet(logger, data, holidays_df, periods, uncertainty_samples, changepoint_prior_scale, seasonality_prior_scale, yearly_seasonality, weekly_seasonality, daily_seasonality, seasonality_mode, result_path):

    """
    Generate order forecast for the next 'periods' days using Prophet.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the historical data.
    - holidays_df (pd.DataFrame): DataFrame with holiday information.
    - periods (int, optional): Number of days to forecast into the future.
    - uncertainty_samples (int, optional): Number of samples for uncertainty.
    - changepoint_prior_scale (float, optional): Parameter controlling the flexibility of the automatic changepoint detection.
    - seasonality_prior_scale (float, optional): Parameter controlling the strength of the seasonality model.
    - yearly_seasonality (bool, optional): Include yearly seasonality.
    - weekly_seasonality (bool, optional): Include weekly seasonality.
    - daily_seasonality (bool, optional): Include daily seasonality.
    - seasonality_mode (str, optional): 'multiplicative' or 'additive' seasonality.

    Returns:
    - forecasted_data (pd.DataFrame): DataFrame containing the rounded forecasted values for the next 'periods' days.
    """

    # Aggregate orders per day
    daily_order_counts = data.groupby('DATE')['TIMESTAMP'].count().reset_index()

    # Rename the columns for clarity
    daily_order_counts.columns = ['ds', 'y']

    # Create and fit the model
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        holidays=holidays_df,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        uncertainty_samples=uncertainty_samples,
        seasonality_mode=seasonality_mode
    )

    model.add_country_holidays(country_name='FI')
    model.fit(daily_order_counts)

    # Create a dataframe with future dates
    future = model.make_future_dataframe(periods=periods)

    # Make predictions
    forecast = model.predict(future)

    # Plot the forecast
    fig = model.plot(forecast)
    plt.title('Order Forecast for the Next 7 Days')
    plt.xlabel('Date')
    plt.ylabel('Order Count')
    # Create the target folder if it doesn't exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Save the image to the specified path
    plt.tight_layout()
    plt.savefig(f"{result_path}/Prophet_Forecasting.png")

    # Extract forecasted values for the next 'periods' days
    forecasted_data = forecast[['ds', 'yhat']].tail(periods)

    # Round the forecasted values
    forecasted_data['rounded_yhat'] = forecasted_data['yhat'].round()

    # Print the rounded forecasted values
    logger.info(f"\nRounded Forecasted Number of Orders for the Next {periods} Days:\n{forecasted_data[['ds', 'rounded_yhat']]}")