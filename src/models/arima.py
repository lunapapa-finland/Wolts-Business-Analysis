import itertools
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import os 
import warnings

def grid_search_arima(train_arima, test_arima):
    """
    Perform grid search for ARIMA hyperparameter tuning.

    Parameters:
    - train_arima (pd.DataFrame): The training data for ARIMA model.
    - test_arima (pd.DataFrame): The testing data for evaluation.

    Returns:
    - best_model (ARIMAResultsWrapper): The best ARIMA model.
    - best_order (tuple): The best (p, d, q) order.
    - best_mse (float): The mean squared error of the best model on the test data.
    """

    p_values = range(0, 7)
    d_values = range(0, 5)
    q_values = range(0, 5)

    best_model = None
    best_mse = float('inf')
    best_order = None

    param_combinations = list(itertools.product(p_values, d_values, q_values))
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

    for params in param_combinations:
        p, d, q = params
        order = (p, d, q)

        model_arima = ARIMA(train_arima, order=order, enforce_stationarity=False, enforce_invertibility=False)
        fit_arima = model_arima.fit()

        forecast_arima = fit_arima.get_forecast(steps=len(test_arima))
        predicted_arima = forecast_arima.predicted_mean

        mse = mean_squared_error(test_arima['ORDER_COUNT'], predicted_arima)

        if mse < best_mse:
            best_mse = mse
            best_model = fit_arima
            best_order = order
    warnings.filterwarnings("default", category=UserWarning, module="statsmodels")

    return best_model, best_order, best_mse

def fit_forecast_plot_evaluate_arima(logger, train, test, data, result_path):
    
    """
    Fits ARIMA model, forecasts, plots results, and returns Mean Squared Error (MSE).

    Parameters:
    - train (pd.DataFrame): The training data.
    - test (pd.DataFrame): The testing data.
    - data (pd.DataFrame): The complete dataset.

    Returns:
    - fit_arima (ARIMAResultsWrapper): The fitted ARIMA model.
    - mse_arima (float): Mean Squared Error of the ARIMA model on the test data.
    """
    logger.info('====================ARIMA==================')
    # Extract the 'ORDER_COUNT' column for ARIMA
    train_arima = train[['DATE', 'ORDER_COUNT']]
    test_arima = test[['DATE', 'ORDER_COUNT']]

    # Set 'DATE' as the index for ARIMA
    train_arima.set_index('DATE', inplace=True)
    test_arima.set_index('DATE', inplace=True)

    # Perform grid search for ARIMA tuning
    fit_arima, order_arima, mse_arima_train = grid_search_arima(train_arima, test_arima)
    logger.info(f'Best Arima Order is {order_arima}')
    logger.info(f"Mean Squared Error (Arima) Train: {mse_arima_train}")

    # Forecast using ARIMA
    forecast_arima = fit_arima.get_forecast(steps=len(test_arima))
    predicted_arima = forecast_arima.predicted_mean

    # Plot ARIMA results
    plt.figure(figsize=(8, 5))
    plt.plot(data['DATE'], data['ORDER_COUNT'], label='Actual')
    plt.plot(test_arima.index, predicted_arima, label='ARIMA Forecast', linestyle='dashed')
    plt.title('ARIMA Forecast')
    plt.legend()
    plt.xticks(rotation=90)  # Rotate x-axis labels

    # Create the target folder if it doesn't exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Save the image to the specified path
    plt.tight_layout()
    plt.savefig(f"{result_path}/Arima_Fit.png")

