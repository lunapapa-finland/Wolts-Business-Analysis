import pandas as pd
from geopy.distance import geodesic

def feature_engineering(data):
    """
    Generate New Features

    Parameters:
    - data (pd.DataFrame): Original DataFrame

    Returns:
    - pd.DataFrame: DataFrame with New Features.
    """

    # Convert TIMESTAMP to datetime format
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])

    # Extract the hour from the timestamp
    data['HOUR_OF_DAY'] = data['TIMESTAMP'].dt.hour

    # Create new features: day of the month, time of day, day of the week
    data['DAY_OF_MONTH'] = data['TIMESTAMP'].dt.day
    data['DATE'] = data['TIMESTAMP'].dt.date
    data['DAY_OF_WEEK'] = data['TIMESTAMP'].dt.dayofweek

    # Create new feature: distance between user and venue locations using ellipsoidal model(‘WGS-84’)
    data['DISTANCE(METERS)'] = data.apply(lambda row: geodesic((row['USER_LAT'], row['USER_LONG']),
                                                  (row['VENUE_LAT'], row['VENUE_LONG'])).km, axis=1)
    return data