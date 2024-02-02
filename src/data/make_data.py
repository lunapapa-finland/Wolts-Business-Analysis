import configparser
import copy

from src.utils.logger import *
from src.utils.configparser import *
from src.utils.utils import *
from src.features.feature_process import *
from src.models.arima import *
from src.models.prophet import *
from datetime import datetime


def make_dataset(logger, parameters):
    # Read  data from data source
    data = read_data_from_github(parameters['raw_link'])
    
    # Perform Feature Engeneering
    data = copy.deepcopy(feature_engineering(data))

    # Perform Descriptive Statistics
    basic_analysis(data, logger)

    # Perform Correlation Analysis
    plot_correlation_heatmap(data, parameters['result_path'], result_name='Raw Data')
    print(f"DEA is Done...\n")

    # Perform Outlier Analysis
    outlier_indices = find_outliers(data, parameters['result_path'])
    logger.info(f"Indices of outliers: {outlier_indices}")
    print(f"Outlier detection is Done...\n")

    # Perform Graphic Visualization
    count_and_plot_histogram(data, 'ACTUAL_DELIVERY_MINUTES - ESTIMATED_DELIVERY_MINUTES', parameters['result_path'])
    count_and_plot_histogram(data, 'ACTUAL_DELIVERY_MINUTES', parameters['result_path'])
    count_and_plot_histogram(data, 'ESTIMATED_DELIVERY_MINUTES', parameters['result_path'])
    scatter_for_est_act(data, parameters['result_path'] )
    distribution_for_est_act(data, parameters['result_path'] )

    # Calculate the percentage distribution
    distribution_calculation(data,logger)
    print(f"ASSESSMENT OF CURRENT ESTIMATED DELIVERY TIME is Done...\n")

    # Create map based on user spatial info. and perform user clustering based on spatial info., order distance and  deviation of delivery time
    user_columns = ['USER_LAT', 'USER_LONG', 'DISTANCE(METERS)', 'ACTUAL_DELIVERY_MINUTES - ESTIMATED_DELIVERY_MINUTES']
    user_map_center = [data['USER_LAT'].mean(), data['USER_LONG'].mean()]
    create_cluster_map(data, user_columns, eps=0.1, min_samples=5, map_center=user_map_center, icon_color='blue', output_filename='user_cluster_map.html', cluster_type='USER_CLUSTER', result_path=parameters['result_path'])

    # Create map based on venue spatial info. and perform venue clustering based on spatial info., order distance and  deviation of delivery time
    venue_columns = ['VENUE_LAT', 'VENUE_LONG', 'DISTANCE(METERS)', 'ACTUAL_DELIVERY_MINUTES - ESTIMATED_DELIVERY_MINUTES']
    venue_map_center = [data['VENUE_LAT'].mean(), data['VENUE_LONG'].mean()]
    create_cluster_map(data, venue_columns, eps=0.1, min_samples=5, map_center=venue_map_center, icon_color='red', output_filename='venue_cluster_map.html', cluster_type='VENUE_CLUSTER', result_path=parameters['result_path'])
    print(f"GEOSPATIAL EXAMINATION THROUGH SPATIALLOCATION is Done...\n")

    # Create Temporal Analysis for USER_CLUSTER
    temporal_analysis_plot(data, cluster_type='USER_CLUSTER', result_path=parameters['result_path'])
    # Create Temporal Analysis for VENUE_CLUSTER
    temporal_analysis_plot(data, cluster_type='VENUE_CLUSTER', result_path=parameters['result_path'])
    # Create Temporal Analysis for all data
    aggregate_temporal_analysis_plot(data, result_path=parameters['result_path'])
    print(f"TEMPORAL INVESTIGATION ALIGNED WITH USER AND VENUE CLUSTER is Done...\n")

    # Aggregate orders per day
    agg_data=order_aggregation(data)
    # Check CORR
    plot_correlation_heatmap(agg_data, result_path=parameters['result_path'], result_name='Aggregated Data')
    # Split data
    train, test = split_data(agg_data)
    # ARIMA
    fit_forecast_plot_evaluate_arima(logger, train, test, agg_data, result_path=parameters['result_path'])

    # Prophet
    # Create holidays DataFrame for the period of 2022-08-01 to 2020-09-30
    holidays_df = pd.DataFrame({
        'ds': ['2020-09-22'],  # Dates of holidays
        'holiday': ['September Equinox'],  # Names or types of holidays
    })
    fit_forecast_plot_evaluate_prophet(logger, train, test, agg_data, holidays_df, country_name = 'FI', result_path=parameters['result_path'])

    # Predict the amount of orders for the coming 7 days
    order_forecast_prophet(logger,data, holidays_df, periods=7, uncertainty_samples=1000, changepoint_prior_scale= 0.01, seasonality_prior_scale= 10.0, yearly_seasonality= False, weekly_seasonality= True, daily_seasonality= True, seasonality_mode = 'additive', result_path=parameters['result_path'])
    print(f"FORECASTING ORDERS is Done...\n")
    print(f"Analysis is done. Please check result in {parameters['log_path']}, and corresponding result in {parameters['result_path']}")

if __name__ == "__main__":

    # Read configuration from file
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Access preprocessing variables and remove comments
    parameters = remove_comments_and_convert(config, 'global')

    # Create a logger
    logger = get_logger('make_data.log', parameters['log_path'])
    logger.info(f'==========New Test at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}==========')
    print(f"Processing, check log later in {parameters['log_path']}\n")

    # Call the make_dataset function with parsed arguments
    make_dataset(logger, parameters)
