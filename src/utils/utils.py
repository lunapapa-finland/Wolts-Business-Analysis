import pandas as pd
import io
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import folium
from folium.plugins import MarkerCluster, HeatMap
from sklearn.cluster import DBSCAN
import copy
import numpy as np

def check_path_existing(path):
    # Create the target folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

def read_data_from_github(raw_link):
    """
    Read CSV data from the GitHub raw link

    Parameters:
    - raw_link (str): github raw data link

    Returns:
    - pd.DataFrame: Original Data in DataFrame Type.
    """

    data = pd.read_csv(raw_link)
    return data

def basic_analysis(data, logger):
    """
    Basic EDA

    Parameters:
    - data: pandas DataFrame

    Returns:
    - None
    """
    # Display basic information about the DataFrame
    with io.StringIO() as buffer, pd.option_context('display.max_rows', None, 'display.max_columns', None):
        data.info(buf=buffer)
        logger.info(f"\n{buffer.getvalue()}")
        
        # Display summary statistics for numerical columns
        summary_statistics = data.describe()
        logger.info(f"\n{summary_statistics.to_string()}")

def plot_correlation_heatmap(data, result_path, result_name):
    """
    Plot a correlation heatmap for the given DataFrame and save it to an image.

    Parameters:
    - data: pandas DataFrame
        The DataFrame for which the correlation heatmap needs to be plotted.
    - result_path: str
        The path to save the correlation heatmap image.

    Returns:
    - None
    """
    correlation_matrix = data.corr(numeric_only=True)

    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap for Raw Data')

    # Create the target folder if it doesn't exist
    check_path_existing(result_path)

    # Save the image to the specified path
    plt.tight_layout()
    plt.savefig(f"{result_path}/Correlation Heatmap for {result_name}.png")

def find_outliers(data, result_path):
    """
    Find the indices of outliers using z-score.

    Parameters:
    - data: pandas DataFrame
        The DataFrame for which outliers need to be detected.

    Returns:
    - list
        A list of indices where outliers are found.
    """
    numeric_columns = data.select_dtypes(include=[float, int]).columns

    # Calculate z-scores only for numeric columns
    z_scores = stats.zscore(data[numeric_columns])
    abs_z_scores = abs(z_scores)
    outliers = (abs_z_scores > 3).all(axis=1)

    # Extract indices where outliers are True
    outlier_indices = data.index[outliers].tolist()

    # Create subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 20))

    # Plot boxplot for the entire dataset
    sns.boxplot(data=data.drop(columns=['DISTANCE(METERS)']), ax=axes[0])
    axes[0].set_title('Boxplot for Small Scale')
    axes[0].set_xticks(axes[0].get_xticks())
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)


    # Plot boxplot for the subset
    sns.boxplot(data=data[['DISTANCE(METERS)']], ax=axes[1])
    axes[1].set_title('Boxplot for Large Scall')
    axes[1].set_xticks(axes[1].get_xticks())
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)

    # Create the target folder if it doesn't exist
    check_path_existing(result_path)

    # Save the image to the specified path
    plt.tight_layout()
    plt.savefig(result_path+"/Outlier Boxplot.png")

    return outlier_indices

def count_and_plot_histogram(data, column, result_path):
    """
    Count and plot histogram for a specific column.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column for which the histogram needs to be plotted.

    Returns:
    - None
    """
    # Plotting histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=30, color='skyblue', edgecolor='black')

    # Set title and labels
    plt.title(f'Histogram of {column}')
    plt.xlabel(f'{column} Value')
    plt.ylabel('Frequency')

    # Add grid for better readability
    plt.grid(True)

    # Create the target folder if it doesn't exist
    check_path_existing(result_path)

    # Save the image to the specified path
    plt.savefig(f"{result_path}/{column}_histogram.png")

def scatter_for_est_act (data, result_path):

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data, x='ESTIMATED_DELIVERY_MINUTES', y='ACTUAL_DELIVERY_MINUTES', alpha=0.1)
    
    # Add a reference line
    plt.plot([0, data['ESTIMATED_DELIVERY_MINUTES'].max()], [0, data['ESTIMATED_DELIVERY_MINUTES'].max()], color='red', linestyle='--')
    
    plt.xlabel('Estimated Delivery Time')
    plt.ylabel('Actual Delivery Time')
    plt.title('Estimated vs. Actual Delivery Time')

    # Create the target folder if it doesn't exist
    check_path_existing(result_path)

    # Save the image to the specified path
    plt.tight_layout()
    plt.savefig(f"{result_path}/Scatter for Est Act Time.png")

def distribution_for_est_act (data, result_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['ESTIMATED_DELIVERY_MINUTES'], kde=True, label='Estimated Delivery Time', color='blue')
    sns.histplot(data['ACTUAL_DELIVERY_MINUTES'], kde=True, label='Actual Delivery Time', color='orange')
    plt.legend()
    plt.xlabel('Delivery Time')
    plt.ylabel('Frequency')
    plt.title('Distribution of Estimated and Actual Delivery Time')

    # Create the target folder if it doesn't exist
    check_path_existing(result_path)

    # Save the image to the specified path
    plt.tight_layout()
    plt.savefig(f"{result_path}/Distribution for Est Act Time.png")

def distribution_calculation(data, logger): 
    # Calculate the percentage distribution
    total_rows = len(data)
    percentage_greater = (data['ACTUAL_DELIVERY_MINUTES'] > data['ESTIMATED_DELIVERY_MINUTES']).sum() / total_rows * 100
    percentage_equal = (data['ACTUAL_DELIVERY_MINUTES'] == data['ESTIMATED_DELIVERY_MINUTES']).sum() / total_rows * 100
    percentage_less = (data['ACTUAL_DELIVERY_MINUTES'] < data['ESTIMATED_DELIVERY_MINUTES']).sum() / total_rows * 100
    logger.info(f"Percentage of actual delivery time greater than estimated: {percentage_greater:.2f}%")
    logger.info(f"Percentage of actual delivery time equal to estimated: {percentage_equal:.2f}%")
    logger.info(f"Percentage of actual delivery time less than estimated: {percentage_less:.2f}%")

def create_cluster_map(data, columns, eps, min_samples, map_center, icon_color, output_filename, cluster_type, result_path):
    """
    Create a map with clusters based on specified columns.

    Parameters:
    - data: pandas DataFrame
        The DataFrame containing the data.
    - columns: list
        The list of columns to use for clustering.
    - eps: float
        The epsilon parameter for DBSCAN.
    - min_samples: int
        The minimum number of samples for DBSCAN.
    - map_center: list
        The center coordinates of the map.
    - icon_color: str
        The color for the cluster markers.
    - output_filename: str
        The filename to save the HTML map.
    - cluster_type: str
        The cluster category.

    Returns:
    - None
    """
    # Select relevant columns for clustering
    cluster_data = data[columns]

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    data[cluster_type] = dbscan.fit_predict(cluster_data)
    temp_data = copy.deepcopy(data)
    # print(data[cluster_type].value_counts().get(-1, 0))

    # Create a map centered at the specified coordinates
    m = folium.Map(location=map_center, zoom_start=12)

    # Create a HeatMap layer based on cluster density with adjusted parameters
    heat_data = [[point[0], point[1]] for point in data[columns].values]
    HeatMap(heat_data, radius=15, blur=13).add_to(m)

    # Add clusters to the map with different colors
    marker_cluster = MarkerCluster().add_to(m)
    for idx, row in temp_data.iterrows():
        popup_text = f"{cluster_type}: {row[cluster_type]}"
        folium.Marker([row[columns[0]], row[columns[1]]], popup=popup_text, icon=folium.Icon(color=icon_color)).add_to(marker_cluster)
    
    # Create the target folder if it doesn't exist
    check_path_existing(result_path)
    # Save the map as an HTML file
    m.save(f"{result_path}/{output_filename}")

def temporal_analysis_plot(data, cluster_type, result_path):
    """
    Create temporal_analysis for clusters.

    Parameters:
    - data: pandas DataFrame
        The DataFrame containing the data.
    - cluster_type: str
        USER_CLUSTER or VENUE CLUSTER.

    Returns:
    - None
    """
    # Define the number of subplots per row
    subplots_per_row = 9
    num_rows = np.ceil(len(data[cluster_type].unique()) / subplots_per_row).astype(int)

    # Create subplots
    fig, axes = plt.subplots(num_rows, subplots_per_row, figsize=(2*subplots_per_row, 2*num_rows), sharex=True, sharey=True)

    # Flatten the axes array for easy indexing
    axes = np.array(axes).flatten()

    for idx, (cluster_label, cluster_data) in enumerate(data.groupby(cluster_type)):
        ax = axes[idx]
        ax.hist(cluster_data['HOUR_OF_DAY'], label=f'{cluster_type} {cluster_label}')
        ax.set_title(f'{cluster_type} {cluster_label}')
        ax.tick_params(axis='x', rotation=90)
        ax.set_xlabel('HOUR_OF_DAY')
        ax.set_ylabel('Order Count')

    # Create the target folder if it doesn't exist
    check_path_existing(result_path)

    # Save the image to the specified path
    plt.tight_layout()
    plt.savefig(f"{result_path}/temporal_analysis_for_cluster_{cluster_type}.png")

def aggregate_temporal_analysis_plot(data, result_path):

    # Group by the 'HOUR_OF_DAY' and count the number of entries in each hour
    aggregated_data = data.groupby('HOUR_OF_DAY').size().reset_index(name='Count')

    # Plot the histogram
    plt.figure(figsize=(12, 6))
    plt.bar(aggregated_data['HOUR_OF_DAY'], aggregated_data['Count'], color='skyblue', edgecolor='black')

    # Set plot labels and title
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.title('Aggregated Count Based on Hour of Day')

    # Create the target folder if it doesn't exist
    check_path_existing(result_path)

    # Save the image to the specified path
    plt.tight_layout()
    plt.savefig(f"{result_path}/temporal_analysis_for_all_data.png")

def order_aggregation(data):
    """
    Aggregate order data by grouping it based on DATE.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the raw order data.

    Returns:
    - agg_data (pd.DataFrame): Aggregated DataFrame with count and median values.
    """
    # Group by DATE and calculate the aggregate count and median
    agg_data = data.groupby('DATE').agg({
        'TIMESTAMP': 'count',
        'WIND_SPEED': 'median',
        'CLOUD_COVERAGE': 'median',
        'TEMPERATURE': 'median',
        'PRECIPITATION': 'median',
        'DAY_OF_WEEK': 'first',
        'DAY_OF_MONTH': 'first',
    }).reset_index()

    # Rename the columns for clarity
    agg_data.columns = ['DATE', 'ORDER_COUNT', 'WIND_SPEED_MEDIAN', 'CLOUD_COVERAGE_MEDIAN',
                        'TEMPERATURE_MEDIAN', 'PRECIPITATION_MEDIAN', 'DAY_OF_WEEK', 'DAY_OF_MONTH']
    return agg_data

def split_data(data, split_ratio=0.8):
    """
    Splits the data into training and testing sets based on the specified ratio.

    Parameters:
    - data (pd.DataFrame): The DataFrame to be split.
    - split_ratio (float, optional): The ratio of the data to be used for training. Default is 0.8.

    Returns:
    - train (pd.DataFrame): The training set.
    - test (pd.DataFrame): The testing set.
    """
    train_size = int(len(data) * split_ratio)
    train, test = data[:train_size], data[train_size:]
    return train, test