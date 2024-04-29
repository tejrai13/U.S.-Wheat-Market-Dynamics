import requests
import pandas as pd

def fetch_weather_data_for_year(year, station_id, datatype, api_key):
    url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'
    records = []
    offset = 1
    while True:
        params = {
            'datasetid': 'GHCND',
            'stationid': station_id,
            'startdate': f'{year}-01-01',
            'enddate': f'{year}-12-31',
            'datatypeid': datatype,
            'units': 'standard',
            'limit': 1000,
            'offset': offset
        }
        headers = {'token': api_key}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                records.extend(data['results'])
                if len(data['results']) < params['limit']:
                    break
                offset += params['limit']
            else:
                print(f"No results in data for {year}")
                break
        else:
            print(f"Failed to retrieve data for {year} from {station_id}: {response.status_code}")
            print(response.json())
            break
    return pd.DataFrame(records)

api_key = 'eWpYkdcSAUMGxpvhzAcsqYKoTWVOLnfm'

# Stations
tavg_station_id = 'GHCND:USR0000KCIM'
precip_station_id = 'GHCND:US1KSAL0001'

years = [2017, 2018, 2019, 2020]  # List of years to fetch data for

# Initialize empty DataFrames to store cumulative data
tavg_cumulative_df = pd.DataFrame()
prcp_cumulative_df = pd.DataFrame()

# Fetch and save data for each year
for year in years:
    # Fetch TAVG data
    tavg_data = fetch_weather_data_for_year(year, tavg_station_id, 'TAVG', api_key)
    tavg_cumulative_df = pd.concat([tavg_cumulative_df, tavg_data], ignore_index=True)
    
    # Fetch PRCP data
    prcp_data = fetch_weather_data_for_year(year, precip_station_id, 'PRCP', api_key)
    prcp_cumulative_df = pd.concat([prcp_cumulative_df, prcp_data], ignore_index=True)

# Save the cumulative data to CSV files
tavg_cumulative_df.to_csv('tavg_kansas_cimarron.csv', index=False)
prcp_cumulative_df.to_csv('precipitation_kansas.csv', index=False)





