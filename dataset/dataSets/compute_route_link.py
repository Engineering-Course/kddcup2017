import math
from datetime import datetime,timedelta
import numpy as np


file_suffix = '.csv'
route_dict = {}
route_link_dict = {}


def create(in_file):

    in_file_name = in_file + file_suffix
    # Step 1: Load trajectories
    fr = open(in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()
    # print(traj_data[0])

    for i in range(len(traj_data)):
        each_traj = traj_data[i].replace('"', '').split(',')
        intersection_id = each_traj[0]
        tollgate_id = each_traj[1]
        route_id = intersection_id + '-' + tollgate_id
        link_length = 0
        for j in xrange(len(each_traj)-2):
           link_id = each_traj[j+2]
           if j == len(each_traj) - 3:
               link_id = link_id[:-1]
           link_length += route_dict[link_id]
           route_link_dict[route_id] = link_length / 2.0
    print route_link_dict

def read_link_data(in_file):

    in_file_name = in_file + file_suffix
    fr = open(in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()

    for i in range(len(traj_data)):
        each_traj = traj_data[i].replace('"', '').split(',')
        # print each_traj
        link_id = each_traj[0]
        length = each_traj[1]
        lanes = each_traj[3]
        route_dict[link_id] = float(length) / float(lanes)

def main():
    link_file = 'training/links (table 3)'
    read_link_data(link_file)
    in_file = 'training/routes (table 4)'
    create(in_file)

if __name__ == '__main__':
    main()

