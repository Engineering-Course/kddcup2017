import math
from datetime import datetime,timedelta
import numpy as np


file_suffix = '.csv'
# route_dict = {'A-2':0, 'A-3':100, 'B-1':200, 'B-3':300, 'C-1':400, 'C-3':500}
route_dict = {'B-3': 165.0, 'B-1': 241.0, 'A-3': 276.66, 'A-2': 111.33, 'C-3': 260.66, 'C-1': 336.66}
weather_dict = {}
direction_sum = []
speed_sum = []
temperature_sum = []
humidity_sum = []
ave_list = []

def create(in_file):

    in_file_name = in_file + file_suffix
    # Step 1: Load trajectories
    fr = open(in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()
    # print(traj_data[0])

    six_time = []
    temp_time = datetime.now()
    temp_travel_time = 0
    count = 0
    for i in xrange(0, len(traj_data), ):
        # print traj_data[i]
        each_traj = traj_data[i].replace('"', '').split(',')
        intersection_id = each_traj[0]
        tollgate_id = each_traj[1]
        interval_left = each_traj[2][1:]
        travel_time = each_traj[4][:-1]
        start_time = datetime.strptime(interval_left, "%Y-%m-%d %H:%M:%S")
        start_time_window = datetime(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, 0)
        # print start_time_window.minute
        if (start_time_window.hour == 6 or start_time_window.hour == 15) and start_time_window.minute == 0:
            six_time = []
            six_time.append(travel_time)
            temp_time = start_time_window
        else:
            if start_time_window.minute - temp_time.minute == 20 and start_time_window.hour == temp_time.hour:
                temp_time = start_time_window
                six_time.append(travel_time)
            else:
                diff_time = start_time_window - temp_time
                diff_window = diff_time.seconds / 1200
                if diff_window <= 6:
                    for dd in xrange(diff_window-1):
                        six_time.append(str(0))
                    six_time.append(travel_time)
                    temp_time = start_time_window
                else:
                    ###  miss end_time 17:00 and 08:00
                    six_time = []
                    six_time.append(str(0))
                    interval1 = abs(start_time_window.hour * 60 + start_time_window.minute - 6 * 60)
                    interval2 = abs(start_time_window.hour * 60 + start_time_window.minute - 15 * 60)
                    diff_time2 = min(interval1, interval2)
                    diff_window2 = diff_time2 / 20
                    for dd in xrange(diff_window2-1):
                        six_time.append(str(0))
                    six_time.append(travel_time)
                    temp_time = start_time_window
        if len(six_time) == 6:
            if (str(0)) in six_time:
                fw = open('testing_six/travel_{}_miss.txt'.format(count), 'w')
            else:
                fw = open('testing_six/travel_{}.txt'.format(count), 'w')
            route_id = route_dict[intersection_id+'-'+tollgate_id]
            obj_time = start_time_window.hour * 60 + start_time_window.minute
            fw.write(str(route_id))
            for tt in six_time:
                fw.write(' ' + tt)
            fw.write(' ' + str(obj_time/4.0))
            weekday = start_time_window.weekday() * 30 + 100
            fw.write(' ' + str(weekday))
            fw.close()
            count += 1

        # print six_time
            # six_time = []


        # wait = raw_input()

def read_weather_data(in_file):

    in_file_name = in_file + file_suffix
    fr = open(in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()

    for i in range(len(traj_data)):
        each_traj = traj_data[i].replace('"', '').split(',')
        # print each_traj
        date = each_traj[0]
        hour = each_traj[1]
        date_time = datetime.strptime(date, "%Y-%m-%d")
        time_window = datetime(date_time.year, date_time.month, date_time.day, int(hour), 0, 0)
        wind_direction = min(float(each_traj[4]), 360.0)
        wind_speed = float(each_traj[5]) * 100
        temperature = float(each_traj[6]) * 10
        rel_humidity = float(each_traj[7]) * 3
        weather_dict[time_window] = [wind_direction, wind_speed, temperature, rel_humidity]
        window_1 = time_window + timedelta(hours=1)
        weather_dict[window_1] = weather_dict[time_window]
        window_2 = time_window + timedelta(hours=2)
        weather_dict[window_2] = weather_dict[time_window]
        direction_sum.append(wind_direction)
        speed_sum.append(wind_speed)
        temperature_sum.append(temperature)
        humidity_sum.append(rel_humidity)
    ave_list.append(np.mean(direction_sum))
    ave_list.append(np.mean(speed_sum))
    ave_list.append(np.mean(temperature_sum))
    ave_list.append(np.mean(humidity_sum))


def main():
    # weather_file = 'training/weather (table 7)_training_update'
    # weather_file = 'testing/weather (table 7)_test1'
    # read_weather_data(weather_file)
    # in_file = 'training/training_20min_avg_travel_time'
    in_file = 'testing/test1_20min_avg_travel_time_miss_endtime'
    create(in_file)

if __name__ == '__main__':
    main()

