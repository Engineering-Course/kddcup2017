import math
from datetime import datetime,timedelta
import numpy as np


file_suffix = '.csv'
route_dict = {'B-3': 165.0, 'B-1': 241.0, 'A-3': 276.66, 'A-2': 111.33, 'C-3': 260.66, 'C-1': 336.66}


def create(in_file, out_file):

    in_file_name = in_file + file_suffix
    out_file_name = out_file + file_suffix
    # Step 1: Load trajectories
    fr = open(in_file_name, 'r')
    fw = open(out_file_name, 'w')
    header = fr.readline()  # skip the header
    fw.write(header)
    traj_data = fr.readlines()

    fr.close()

    temp_time = datetime.now()
    for i in range(len(traj_data)):
        # print traj_data[i]
        each_traj = traj_data[i].replace('"', '').split(',')
        intersection_id = each_traj[0]
        tollgate_id = each_traj[1]
        interval_left = each_traj[2][1:]
        travel_time = each_traj[4][:-1]
        start_time = datetime.strptime(interval_left, "%Y-%m-%d %H:%M:%S")
        start_time_window = datetime(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, 0)

        if start_time_window.day != temp_time.day:
            going_time = datetime(start_time.year, start_time.month, start_time.day, 0, 0, 0)
        if start_time_window.hour == going_time.hour and start_time_window.minute == going_time.minute and start_time_window.day == going_time.day:
            time_window_end = start_time_window + timedelta(minutes=20)
            out_line = ','.join(['"' + intersection_id + '"', '"' + tollgate_id + '"',
                     '"[' + str(start_time_window) + ',' + str(time_window_end) + ')"', '"' + str(travel_time) + '"']) + '\n'
            fw.writelines(out_line)
        else:
            while start_time_window != going_time:
                time_window_end = going_time + timedelta(minutes=20)
                out_line = ','.join(['"' + intersection_id + '"', '"' + tollgate_id + '"',
                         '"[' + str(going_time) + ',' + str(time_window_end) + ')"', '"' + str(0) + '"']) + '\n'
                fw.writelines(out_line)
                going_time = going_time + timedelta(minutes=20)

            time_window_end = start_time_window + timedelta(minutes=20)
            out_line = ','.join(['"' + intersection_id + '"', '"' + tollgate_id + '"',
                     '"[' + str(start_time_window) + ',' + str(time_window_end) + ')"', '"' + str(travel_time) + '"']) + '\n'
            fw.writelines(out_line)
        going_time = going_time + timedelta(minutes=20)
        temp_time = start_time_window
        # wait = raw_input()

    fw.close()



def main():

    in_file = 'training/training-all_20min_avg_travel_time'
    out_file = 'training/training-all_fill_travel_time'
    create(in_file, out_file)

if __name__ == '__main__':
    main()

