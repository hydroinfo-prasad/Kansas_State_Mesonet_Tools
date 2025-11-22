##    Function that retrieves data from Kansas Mesonet https://mesonet.k-state.edu/.
##    The metadata is provided in http://mesonet.k-state.edu/rest/variables/

import numpy as np
import pandas as pd

import pandas as pd
import requests
from io import StringIO
from timezonefinder import TimezoneFinder
import pytz

def download_weather_ks_mesonet_all(stations, interval, start_date, end_date, lat=None, lon=None, save_file=False):
    """
    Downloads Kansas Mesonet data and optionally converts timezones if coordinates are provided.
    
    Parameters:
    stations (list): List of station names (e.g., ['Manhattan'])
    interval (str): Data interval (e.g., 'day', 'hour')
    start_date (str): Start date string
    end_date (str): End date string
    lat (float, optional): Latitude of the station for timezone conversion.
    lon (float, optional): Longitude of the station for timezone conversion.
    save_file (bool): If True, saves the data to a CSV file.
    """
    
    fmt = "%Y%m%d%H%M%S" 
    stations_str = ','.join(stations)

    # 1. Format dates for the API
    try:
        t_start = pd.to_datetime(start_date).strftime(fmt)
        t_end = pd.to_datetime(end_date).strftime(fmt)
    except ValueError as e:
        print(f"❌ Date Format Error: {e}")
        return pd.DataFrame()

    # 2. Construct URL and Fetch Data
    url = f"http://mesonet.k-state.edu/rest/stationdata/?stn={stations_str}&int={interval}&t_start={t_start}&t_end={t_end}"
    url = url.replace(' ', '%20')

    try:
        response = requests.get(url)
        response.raise_for_status() 

        if "Error: limit of" in response.text:
            print(f"‼️ API Error: Request limit exceeded.")
            return pd.DataFrame()

        # Read CSV
        df = pd.read_csv(
            StringIO(response.text),
            parse_dates=['TIMESTAMP'],
            na_values=['M', 'T', 'n/a']
        )

        if df.empty:
            print("Warning: Downloaded data is empty.")
            return df
            
        print(f"Successfully downloaded {len(df)} records.")

        # 3. Optional Timezone Conversion (Only runs if lat/lon are provided)
        if lat is not None and lon is not None:
            try:
                tf = TimezoneFinder()
                timezone_str = tf.timezone_at(lng=lon, lat=lat)
                
                if timezone_str:
                    print(f"🌍 Detected Timezone: {timezone_str}")
                    local_tz = pytz.timezone(timezone_str)
                    
                    # Ensure TIMESTAMP is UTC-aware before converting
                    if df['TIMESTAMP'].dt.tz is None:
                        df['TIMESTAMP'] = df['TIMESTAMP'].dt.tz_localize('UTC')
                    
                    # Create the new Local Time column
                    df['TIMESTAMP_Local'] = df['TIMESTAMP'].dt.tz_convert(local_tz)
                else:
                    print("Could not determine timezone from coordinates.")
            except Exception as e:
                print(f"Timezone conversion failed: {e}")

        # 4. Save to File (Optional)
        if save_file:
            filename = f"mesonet_{stations_str}_{t_start}_{t_end}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved file to: {filename}")

        return df

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return pd.DataFrame()
