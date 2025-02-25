from sklearn.metrics import root_mean_squared_log_error
import numpy as np
import json
import math

with open("gearformer_search/data/output_EDA.json", 'r') as file:
    data = json.load(file)

N = 30

input_motion_type_correct = 0
output_motion_type_correct = 0
output_motion_vector_xyz_correct = 0
output_motion_vector_sign_correct = 0
output_position_euc_dist = 0
speed_msle = 0
speed_mape = 0
weight = 0

for d in data["input_motion_type"]:
    if d[0] == d[1]:
        input_motion_type_correct += 1

for d in data["output_motion_type"]:
    if d[0] == d[1]:
        output_motion_type_correct += 1

for d in data["output_motion_vector"]:
    if np.dot(d[0], d[1]) == 1:
        output_motion_vector_xyz_correct += 1
        output_motion_vector_sign_correct += 1
    elif np.dot(d[0], d[1]) == -1:
        continue

a = []
for d in data["output_position"]:
    output_position_euc_dist += math.dist(d[0], d[1])
    a.append(math.dist(d[0], d[1]))

a = []
for d in data["output_motion_speed"]:
    speed_msle += root_mean_squared_log_error([d[0]], [d[1]])
    a.append(root_mean_squared_log_error([d[0]], [d[1]]))


a = []
for d in data["weight"]:
    weight += d
    a.append(d)


print("input_motion_type_correct: ", input_motion_type_correct/N)
print("output_motion_type_correct: ", output_motion_type_correct/N)
print("output_motion_vector_xyz: ", output_motion_vector_xyz_correct/N)
print("output_motion_vector_sign_correct: ", output_motion_vector_sign_correct/N)
print("output_position_euc_dist: ", output_position_euc_dist/N)
print("speed_msle: ", speed_msle/N)
print("weight: ", weight/N)
