import math
from datetime import datetime,timedelta
import numpy as np

travel_times = {}

def load_result(in_file):

    in_file_name = in_file
    fr = open(in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()

    for i in xrange(len(traj_data)):
        each_traj = traj_data[i].replace('"', '').split(',')
        intersection_id = each_traj[0]
        tollgate_id = each_traj[1]
        interval_left = each_traj[2][1:]
        start_time = datetime.strptime(interval_left, "%Y-%m-%d %H:%M:%S")
        start_time_window = datetime(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, 0)
        t_time = float(each_traj[-1])

        route_id = intersection_id + '-' + tollgate_id
        if route_id not in travel_times.keys():
            travel_times[route_id] = {}

        if start_time_window not in travel_times[route_id].keys():
            travel_times[route_id][start_time_window] = [t_time]
        else:
            travel_times[route_id][start_time_window].append(t_time)


def create(example_file,out_file_name):

    in_file_name = example_file
    fr = open(in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()


    fw = open(out_file_name, 'w')
    fw.writelines(','.join(['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time']) + '\n')

    for i in xrange(len(traj_data)):
        each_traj = traj_data[i].replace('"', '').split(',')
        intersection_id = each_traj[0]
        tollgate_id = each_traj[1]
        interval_left = each_traj[2][1:]
        start_time = datetime.strptime(interval_left, "%Y-%m-%d %H:%M:%S")
        start_time_window = datetime(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, 0)
        end_time_window = start_time_window + timedelta(minutes=20)
        
        route_id = intersection_id + '-' + tollgate_id
        t_time = travel_times[route_id][start_time_window]

        out_line = ','.join([intersection_id, tollgate_id, '"[' + str(start_time_window) + ',' + str(end_time_window) + ')"', str(t_time[0])]) + '\n'
        fw.writelines(out_line)

    fw.close()
        


def main():
    in_file = 'my_travel_result.csv'
    example_file = 'dataset/submission_sample_travelTime.csv'
    out_file_name = 'submit_travel_result.csv'
    load_result(in_file)
    create(example_file,out_file_name)


if __name__ == '__main__':
    main()

