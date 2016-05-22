import pandas as pd
import time as tlib


cluster_map_path = 'season_1/training_data/cluster_map/cluster_map'
weather_path = 'season_1/training_data/weather_data/weather_data_2016-01-'
traffic_path = 'season_1/training_data/traffic_data/traffic_data_2016-01-'
poi_path = 'season_1/training_data/poi_data/poi_data'
order_path = 'season_1/training_data/order_data/order_data_2016-01-'

cluster_map_data = pd.read_csv(cluster_map_path)
dic_districts = cluster_map_data.set_index('Hash')['Value'].to_dict()

for i in range(1,2):
    if i < 10:
        path = order_path + '0' + str(i)
    else:
        path = order_path + str(i)

    order_data = pd.read_csv(path)
    # Change Hash codes for district number
    order_data['Start_district_hash'].replace(dic_districts, inplace=True)
    order_data['Dest_district_hash'].replace(dic_districts, inplace=True)

    # TODO: Iterate through all time segments 1:144, count nulls in 'Driver_id' -> equivalent to gapij
    for t_segment in range(1,145):
        records = order_data['Driver_id'][order_data['Segment'] == t_segment]
        demand = records.size
        gap = demand - records.count()
        print demand, gap




# order_data_path_root = 'season_1/training_data/order_data/order_data_2016-01-01'

# data = pd.read_csv(order_data_path_root)









