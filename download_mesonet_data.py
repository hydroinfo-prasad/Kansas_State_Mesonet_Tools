import numpy as np
import pandas as pd

def download_weather_ks_mesonet_all(stations, interval, start_date, end_date):
    """
    Function that retrieves data from Kansas Mesonet https://mesonet.k-state.edu/.
    The metadata is provided in http://mesonet.k-state.edu/rest/variables/
    
    Parameters:
    stations (list): List of station names (e.g., ['Manhattan'])
    interval (str): Data interval (e.g., 'day', 'hour')
    start_date (str): Start date string
    end_date (str): End date string
    """
    
    fmt = "%Y%m%d%H%M%S"  # Format required by the Kansas Mesonet
    
    # Process inputs
    stations_str = ','.join(stations)
    start_date_fmt = pd.to_datetime(start_date).strftime(fmt)
    end_date_fmt = pd.to_datetime(end_date).strftime(fmt)
    
    # Construct URL
    url = f"http://mesonet.k-state.edu/rest/stationdata/?stn={stations_str}&int={interval}&t_start={start_date_fmt}&t_end={end_date_fmt}"
    url = url.replace(' ', '%20')

    # Fetch data
    try:
        df = pd.read_csv(url, parse_dates=['TIMESTAMP'])
        return df
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None
