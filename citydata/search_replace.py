import pandas as pd
from tempfile import mkstemp
from shutil import move
from os import remove, close
import time as tlib

order_data_path_root = 'season_1/training_data/order_data/order_data_2016-01-'

def time_to_segment(input_string):
    date, time = input_string.split(' ',1)
    dt = tlib.strptime(input_string, "%Y-%m-%d %H:%M:%S")
    segment = (dt.tm_hour * 6) + (dt.tm_min/10)+1
    return [date, time, segment]

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with open(abs_path,'w') as new_file:
        new_file.write('Order_id,Driver_id,Passenger_id,Start_district_hash,Dest_district_hash,Price,Date,Time,Segment\n')
        with open(file_path) as old_file:
            for line in old_file:
                others, time_string = line.rsplit(',',1)
                time_string = time_string.split('\n')[0]
                if time_string == 'time':
                    continue
                else:
                    date, time, segment = time_to_segment(time_string)
                    to_write = others + ',' + date + ',' + time + ',' + str(segment) + '\n'
                    new_file.write(to_write)
                    # new_file.write(line.replace(pattern, subst))
    close(fh)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

for i in range(1,22):
    if i < 10:
        path = order_data_path_root + '0' + str(i)
    else:
        path = order_data_path_root + str(i)

    replace(path, '\t', ',')





