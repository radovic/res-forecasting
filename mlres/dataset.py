import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class WeatherEnergyDataset(Dataset):
    def __init__(self, 
                 data,
                 target_idx, 
                 window_size=288, # 3 days * 24 hours * 4 quarters
                 horizon=96, # 1 day * 24 hours * 4 quarters
                 device='cpu'
                 ):
        self.data = data # pandas DataFrame
        self.window_size = window_size
        self.horizon = horizon
        self.input_columns = list(self.data.columns)
        self.output_columns = self.data.columns[target_idx]
        self.device = device

    def __len__(self):
        return int(len(self.data) - self.window_size - self.horizon + 1)
    
    def __getitem__(self, idx):
        # Return num_locations * window_size input and num_locations * horizon output
        return torch.Tensor(self.data.iloc[idx:(idx+self.window_size)][self.input_columns].values).to(self.device),\
            torch.Tensor(self.data.iloc[(idx+self.window_size):(idx+self.window_size+self.horizon)][self.output_columns].values).to(self.device)

class GermanWeatherEnergy:
    def __init__(self,
                 val_size=0.2,
                 target_idx=1202,
                 window_size=288, # 3 days * 24 hours * 4 quarters
                 horizon=96, # 1 day * 24 hours * 4 quarters
                 data_dir = None,
                 device='cpu'
                 ):
        ## Load and preprocess data
        self.data_dir = data_dir
        self.val_size = val_size
        print('Loading and preprocessing data...')
        self.generate_dataset()
        print('Processed data generated successfully.')
        self.target_idx = target_idx
        self.window_size = window_size
        self.horizon = horizon
        self.device = device

    def get_train_and_val_data(self):
        return WeatherEnergyDataset(self.train_data.drop(columns=['time']).astype('float32'), self.target_idx, self.window_size, self.horizon, self.device),\
            WeatherEnergyDataset(self.val_data.drop(columns=['time']).astype('float32'), self.target_idx, self.window_size, self.horizon, self.device)

    def get_test_data(self):
        return WeatherEnergyDataset(self.test_data.drop(columns=['time']).astype('float32'), self.target_idx, self.window_size, self.horizon, self.device)

    def generate_dataset(self):
        def fill_missing_weather_data(weather_data):
            weather_data['time_tmp'] = weather_data['time']
            weather_data.set_index(['time', 'longitude', 'latitude'], inplace=True)
            weather_data.reset_index(inplace=True)
            weather_data.set_index('time', inplace=True)
            weather_data = weather_data.groupby(['latitude', 'longitude']).apply(lambda x: x.resample('15min').interpolate())
            weather_data = weather_data.rename(columns={'time_tmp': 'time'})
            weather_data.reset_index(inplace=True, drop=True)
            # Set current data point to contain 24 hours ahead weather data
            previous_start = weather_data['time'].min()
            weather_data['time'] = weather_data['time'] - pd.Timedelta(days=1)
            weather_data = weather_data[weather_data['time'] >= previous_start]
            return weather_data
        
        def get_adjusted_price_differences(prices):
            # Load the HICP data
            hicp = pd.read_csv('/home/martius-lab/Desktop/renewable_ml/data/energy_data/hicp_de.csv')
            hicp['time'] = pd.to_datetime(hicp['DATE'], format='%Y-%m-%d')
            hicp.set_index('time', inplace=True)
            hicp = hicp.resample('1h').asfreq()
            hicp['HICP - Electricity (ICP.M.DE.N.045100.4.INX)'] = hicp['HICP - Electricity (ICP.M.DE.N.045100.4.INX)'].interpolate()
            hicp = hicp[hicp.index >= pd.to_datetime('01-01-2019 00:00', format='%d-%m-%Y %H:%M')]
            hicp = hicp[hicp.index <= pd.to_datetime('31-12-2022 23:00', format='%d-%m-%Y %H:%M')]
            # Calculate the adjusted price differences
            tmp = prices['price'].to_numpy()[1:] - prices['price'].to_numpy()[:-1]
            tmp = np.append(tmp, tmp[-1])
            prices['price'] = tmp / hicp['HICP - Electricity (ICP.M.DE.N.045100.4.INX)'].to_numpy()
            return prices

        def fill_missing_prices_data(prices_data):
            prices_data.set_index('time', inplace=True)
            for column in prices_data.columns:
                if prices_data[column].dtype == 'object':
                    prices_data[column] = prices_data[column].str.replace(',', '.')
                    prices_data[column] = pd.to_numeric(prices_data[column], errors='coerce')
            prices_data = get_adjusted_price_differences(prices_data)
            prices_data = prices_data.resample('15min').asfreq()
            prices_data = prices_data.interpolate(method='linear')
            prices_data = prices_data.reset_index()
            return prices_data

        def fill_missing_realcap_data(realcap_data):
            realcap_data.set_index('time', inplace=True)
            realcap_data = realcap_data[~realcap_data.index.duplicated(keep='first')]
            for column in realcap_data.columns:
                if realcap_data[column].dtype == 'object':
                    realcap_data[column] = realcap_data[column].str.replace('.', '').str.replace(',', '.')
                    realcap_data[column] = pd.to_numeric(realcap_data[column], errors='coerce')
            start_date = '2018-12-31 23:00:00'
            end_date = '2022-12-31 22:45:00'
            new_index = pd.date_range(start=start_date, end=end_date, freq='15min')
            realcap_data = realcap_data.reindex(new_index).ffill()
            realcap_data = realcap_data.reset_index()
            realcap_data.rename(columns={'index': 'time'}, inplace=True)
            return realcap_data

        def convert_to_UTC(data, source='Europe/Berlin'):
            data['time'] = data['time'].dt.tz_localize(source, ambiguous='NaT').dt.tz_convert('UTC')
            data['time'] = data['time'].dt.tz_localize(None)
            data['time'] = data['time'].interpolate()
            return data
        
        def split_time_into_integers(data):
            data['year'] = data['time'].dt.year
            data['month_cos'] = np.round(np.cos(data['time'].dt.month / 12 * 2 * np.pi), 2)
            data['month_sin'] = np.round(np.sin(data['time'].dt.month / 12 * 2 * np.pi), 2)
            data['day_cos'] = np.round(np.cos(data['time'].dt.day / 31 * 2 * np.pi), 2)
            data['day_sin'] = np.round(np.sin(data['time'].dt.day / 31 * 2 * np.pi), 2)
            data['hour_cos'] = np.round(np.cos(data['time'].dt.hour / 24 * 2 * np.pi), 2)
            data['hour_sin'] = np.round(np.sin(data['time'].dt.hour / 24 * 2 * np.pi), 2)
            data['minute_cos'] = np.round(np.cos(data['time'].dt.minute / 60 * 2 * np.pi), 2)
            data['minute_sin'] = np.round(np.sin(data['time'].dt.minute / 60 * 2 * np.pi), 2)
            return data
            
        ## Load the weather data
        start_time = time.time()
        weather_data19 = pd.read_csv(os.path.join(self.data_dir, "weather_data_19-21_de.csv")).drop(columns=['forecast_origin']) # to be used as training/validation data
        weather_data22 = pd.read_csv(os.path.join(self.data_dir, "weather_data_22_de.csv")).drop(columns=['forecast_origin']) # to be used as test data
        print(f"Loaded weather data in {time.time() - start_time:.2f} seconds.")

        ## Load the energy data
        start_time = time.time()
        # Prices data of length (365 * 4 + 1) * 24 = 35064
        prices_data = pd.read_csv(os.path.join(self.data_dir, "energy_data/prices_eu.csv"), delimiter=';')
        prices_data = prices_data[['Date from', 'Germany/Luxembourg [€/MWh]']].rename(columns={'Date from': 'time', 'Germany/Luxembourg [€/MWh]': 'price'})
        # Capacities data of length 4
        installed_capacity_data = pd.read_csv(os.path.join(self.data_dir, 'energy_data/installed_capacity_de.csv'), delimiter=';').rename(columns={'Date from': 'time'})
        installed_capacity_data = installed_capacity_data[['time', 'Wind Offshore [MW] ', 'Wind Onshore [MW]', 'Photovoltaic [MW]']]
        installed_capacity_data.rename(columns={'Wind Offshore [MW] ': 'wind_offshore_capacity', 
                                                'Wind Onshore [MW]': 'wind_onshore_capacity', 
                                                'Photovoltaic [MW]': 'photovoltaic_capacity'}, inplace=True)
        # Realised supply of length (365 * 4 + 1) * 24 * 4 = 140256
        realised_supply_data = pd.read_csv(os.path.join(self.data_dir, 'energy_data/realised_supply_de.csv'), delimiter=';').rename(columns={'Date from': 'time'})
        realised_supply_data = realised_supply_data[['time', 'Wind Offshore [MW] ', 'Wind Onshore [MW]', 'Photovoltaic [MW]']]
        realised_supply_data.rename(columns={'Wind Offshore [MW] ': 'wind_offshore_supply',
                                            'Wind Onshore [MW]': 'wind_onshore_supply',
                                            'Photovoltaic [MW]': 'photovoltaic_supply'}, inplace=True)
        # Realised demand of length (365 * 4 + 1) * 24 * 4 = 140256
        realised_demand_data = pd.read_csv(os.path.join(self.data_dir, 'energy_data/realised_demand_de.csv'), delimiter=';').rename(columns={'Date from': 'time'})
        print(f"Loaded energy data in {time.time() - start_time:.2f} seconds.")

        ## Merge data
        start_time = time.time()
        realised_demand_data = realised_demand_data.drop(columns=['time', 'Date to'])
        realisation_data = pd.merge(realised_supply_data, realised_demand_data, left_index=True, right_index=True)
        print(f"Merged realisation data in {time.time() - start_time:.2f} seconds.")

        # Convert to datetime format
        start_time = time.time()
        weather_data19['time'] = pd.to_datetime(weather_data19['time'], format='%Y-%m-%d %H:%M:%S')
        weather_data22['time'] = pd.to_datetime(weather_data22['time'], format='%Y-%m-%d %H:%M:%S')
        realisation_data['time'] = pd.to_datetime(realisation_data['time'], format='%d.%m.%y %H:%M')
        installed_capacity_data['time'] = pd.to_datetime(installed_capacity_data['time'], format='%d.%m.%y')
        prices_data['time'] = pd.to_datetime(prices_data['time'], format='%d.%m.%y %H:%M')
        print(f"Converted to datetime format in {time.time() - start_time:.2f} seconds.")
        
        # Convert energy data to UTC
        start_time = time.time()
        prices_data = convert_to_UTC(prices_data)
        realisation_data = convert_to_UTC(realisation_data)
        installed_capacity_data = convert_to_UTC(installed_capacity_data)
        print(f"Converted energy data to UTC in {time.time() - start_time:.2f} seconds.")

        # Extend weather data with missing values
        start_time = time.time()
        weather_data19 = fill_missing_weather_data(weather_data19)
        weather_data22 = fill_missing_weather_data(weather_data22)
        print(f"Filled missing weather data in {time.time() - start_time:.2f} seconds.")    

        # Stack locations
        start_time = time.time()
        weather_data19.drop(columns=['latitude', 'longitude'], inplace=True)
        weather_data22.drop(columns=['latitude', 'longitude'], inplace=True)
        tmp_wd19, tmp_wd19_steps = None, int(len(weather_data19) / 80)
        tmp_wd22, tmp_wd22_steps = None, int(len(weather_data22) / 80)
        for idx in range(80):
            tmp19 = weather_data19[idx * tmp_wd19_steps:(idx + 1) * tmp_wd19_steps].rename(columns=lambda x: f"{x}_loc{idx}" if x != 'time' else x)
            tmp22 = weather_data22[idx * tmp_wd22_steps:(idx + 1) * tmp_wd22_steps].rename(columns=lambda x: f"{x}_loc{idx}" if x != 'time' else x)
            if tmp_wd19 is None:
                tmp_wd19 = tmp19
                tmp_wd22 = tmp22
            else:
                tmp_wd19 = pd.merge(tmp_wd19, tmp19, on='time')
                tmp_wd22 = pd.merge(tmp_wd22, tmp22, on='time')
        weather_data19 = tmp_wd19
        weather_data22 = tmp_wd22
        print(f"Stacked locations in {time.time() - start_time:.2f} seconds.")

        # Extend energy data with missing values and merge them
        start_time = time.time()
        realisation_data = fill_missing_realcap_data(realisation_data)
        prices_data = fill_missing_prices_data(prices_data)
        installed_capacity_data = fill_missing_realcap_data(installed_capacity_data)
        energy_data = pd.merge(pd.merge(realisation_data, installed_capacity_data, on='time') , prices_data,how='left', on='time')
        energy_data['price'] = energy_data['price'].interpolate()
        print(f"Filled missing energy data in {time.time() - start_time:.2f} seconds.")

        # Combine the weather data with the energy data
        start_time = time.time()
        trainval_data = pd.merge(weather_data19, energy_data, how='left', on='time')
        self.test_data = pd.merge(weather_data22, energy_data, how='left', on='time').ffill()
        print(f"Combined weather and energy data in {time.time() - start_time:.2f} seconds.")

        # Split time into integers
        start_time = time.time()
        trainval_data = split_time_into_integers(trainval_data)
        self.test_data = split_time_into_integers(self.test_data)
        print(f"Split time into integers in {time.time() - start_time:.2f} seconds.")

        # Split the data into training and validation sets
        start_time = time.time()
        time_split = trainval_data['time'].min() + (trainval_data['time'].max() - trainval_data['time'].min()) * (1 - self.val_size)
        index_split = trainval_data.index[trainval_data['time'] <= time_split][-1] + 1  # Add 1 to include the last element of train
        self.train_data = trainval_data.iloc[:index_split, :].copy()
        self.val_data = trainval_data.iloc[index_split:, :].copy()
        print(f"Split data into training and validation sets in {time.time() - start_time:.2f} seconds.")

        # Normalize the data
        start_time = time.time()
        exclude_columns = {'time', 'wind_offshore_supply', 'wind_onshore_supply', 'photovoltaic_supply'}
        target_columns = set(self.train_data.columns[-9:])
        columns_to_be_normalized = set(self.train_data.columns) - exclude_columns - target_columns
        columns_to_be_normalized = list(columns_to_be_normalized)
        std_dev = self.train_data[columns_to_be_normalized].std()
        columns_to_be_normalized = [col for col in columns_to_be_normalized if std_dev[col] > 1.e-5]
        mean = self.train_data[columns_to_be_normalized].mean()
        std = self.train_data[columns_to_be_normalized].std()
        self.train_data.loc[:, columns_to_be_normalized] = (self.train_data[columns_to_be_normalized] - mean) / std
        self.val_data.loc[:, columns_to_be_normalized] = (self.val_data[columns_to_be_normalized] - mean) / std
        self.test_data.loc[:, columns_to_be_normalized] = (self.test_data[columns_to_be_normalized] - mean) / std
        print(f"Normalized data in {time.time() - start_time:.2f} seconds.")

        '''
        # Save the data
        print("Saving preprocessed data...")
        os.makedirs(os.path.join(self.data_dir, 'preprocessed_data'), exist_ok=True)
        self.trainval_data.to_csv(os.path.join(self.data_dir, 'preprocessed_data/trainval_data.csv'), index=False)
        self.test_data.to_csv(os.path.join(self.data_dir, 'preprocessed_data/test_data.csv'), index=False)
        print("Preprocessed data saved successfully.")
        '''
