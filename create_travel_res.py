import math
from datetime import datetime,timedelta
import numpy as np


file_suffix = '.csv'

def create(in_file, out_file_name):

    in_file_name = in_file + file_suffix
    # Step 1: Load trajectories
    fr = open(in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()
    # print(traj_data[0])

    fw = open(out_file_name, 'w')
    fw.writelines(','.join(['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time']) + '\n')


    count = 0
    for i in xrange(0, len(traj_data), 6):
        # print traj_data[i]
        each_traj = traj_data[i].replace('"', '').split(',')
        intersection_id = each_traj[0]
        tollgate_id = each_traj[1]
        interval_left = each_traj[2][1:]
        start_time = datetime.strptime(interval_left, "%Y-%m-%d %H:%M:%S")
        start_time_window = datetime(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, 0)
        start_time_window = start_time_window + timedelta(minutes=120)

        with open('result/{}.txt'.format(count), 'r') as fi:
            res = fi.readlines()
            res = res[0].split(' ')
        for jj in xrange(6):
            end_time_window = start_time_window + timedelta(minutes=20)
            out_line = ','.join([intersection_id, tollgate_id, '"[' + str(start_time_window) + ',' + str(end_time_window) + ')"', str(res[jj])]) + '\n'
    
            fw.writelines(out_line)
            start_time_window = end_time_window
        count += 1
    fw.close()


def main():
    in_file = 'dataset/dataSets/testing/test1_20min_avg_travel_time_update'
    out_file_name = 'my_travel_result.csv'
    create(in_file, out_file_name)

if __name__ == '__main__':
    main()

