from gear_train_simulator import Simulator
from concurrent.futures import ThreadPoolExecutor
from util.suppress_print import SuppressPrint
import os
import json
import csv
from sklearn.metrics.pairwise import cosine_similarity
simulator = Simulator()
import math
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_log_error
import numpy as np


def json_reader(filename):
    with open(filename) as f:
        data=json.load(f)
    return data

def is_grammatically_correct(language_path, seq):
    """
    input
    ----
    seq: a sequence of tokens

    return
    ----
    True: if this sequence respects the grammar
    False: if this sequence does not respect the grammar
    """
    language = json_reader(language_path)
    grammar = {}
    vocab = {}
    for i in language["vocab"]:
        for j in language["vocab"][i]:
            vocab[j] = i
    vocab["<start>"] = "<start>"
    vocab["<end>"] = "<end>"

    for i in language['grammar']:
        grammar[i["LHS"]] = i["RHS"]

    j = 0
    while(j<len(seq)-1):
        try:
            if [vocab[seq[j+1]]] in grammar[vocab[seq[j]]]:
                j = j+1
            elif [vocab[seq[j+1]], vocab[seq[j+2]]] in grammar[vocab[seq[j]]]:
                j = j+2
            elif [vocab[seq[j+1]], vocab[seq[j+2]], vocab[seq[j+3]]] in grammar[vocab[seq[j]]]:
                j = j+3
            else:
                return False
        except:
            return False
    return True
    
position_euclidian_average = 0
total = 0
weight_sum = 0
input_tr = 0
output_tr = 0
msle = 0
std_pos = []
std_weight = []
std_speed = []
xyz_and_sign = 0

import argparse

max_speed = 216000.0
parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, default="val_dummy.csv", help='path to the csv')
args = parser.parse_args()
csv_file = args.csv_path
with open(csv_file, 'r') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        # xyz = 0
        ccw_gen = 1
        
        r_t = [float(row[0].replace(", device='cuda:0')", "").replace("tensor(", "")), float(row[1].replace(", device='cuda:0')", "").replace("tensor(", ""))]
        ratio = float(row[2].replace(", device='cuda:0')", "").replace("tensor(", ""))
        out_position = [float(row[3].replace(", device='cuda:0')", "").replace("tensor(", "")), float(row[4].replace(", device='cuda:0')", "").replace("tensor(", "")), float(row[5].replace(", device='cuda:0')", "").replace("tensor(", ""))]
        rot =  [float(row[6].replace(", device='cuda:0')", "").replace("tensor(", ""))]
        motion_sign = float(row[7].replace(", device='cuda:0')", "").replace("tensor(", ""))
        
        seq = row[-1][2:-2].split("', '")

        if seq[-1] != '<end>':
            seq.append('<end>')
        id = 0
        try:        
            input_data = {
                        "gear_train_sequence": list(['<start>'] + seq)
                        }
            input_data["id"] = id

            if is_grammatically_correct("language.json", input_data["gear_train_sequence"]):
                total += 1
                with SuppressPrint():
                    res = simulator.run(input_data)
                id += 1
                
                for i in range(3):
                    if res["output_motion_vector"][i] > 0.5 : 
                        res["output_motion_vector"][i] = 1
                        xyz = i
                    elif res["output_motion_vector"][i] < -0.5: 
                        res["output_motion_vector"][i] = -1
                        xyz = i
                        ccw_gen = -1
                    else: res["output_motion_vector"][i] = 0
                
                
                if rot[0] == xyz and ccw_gen == motion_sign:
                    xyz_and_sign += 1
                

                weight_sum += res["weight"]
                                
                msle += root_mean_squared_log_error([ratio], [res["output_motion_speed"]])
                
                euclidean = math.dist(res["output_position"], out_position)
                position_euclidian_average += euclidean


                # evaluate motion type
                if res["input_motion_type"] == "R": res["input_motion_type"] = 0
                else: res["input_motion_type"] = 1

                if res["output_motion_type"] == "R": res["output_motion_type"] = 0
                else: res["output_motion_type"] = 1

                if res["input_motion_type"] == r_t[0]:
                    input_tr += 1

                if res["output_motion_type"] == r_t[1]:
                    output_tr += 1

                std_weight.append(res["weight"])
                std_speed.append(root_mean_squared_log_error([ratio], [res["output_motion_speed"]]))
                std_pos.append(euclidean)


        except EOFError:
          error
          break
    

        
        

print( "file name:", csv_file, 
      "     total:", total, 
      "     position_euclidian_average", position_euclidian_average/total, 
      "     average weight", weight_sum/total,
      "     input_tr:", input_tr/total, "    output_tr:", output_tr/total,
      "     msle", msle/total,
      "     xyz_and_sign ", xyz_and_sign/total )



print("weight", np.average(std_weight), np.std(std_weight))
print("speed", np.average(std_speed), np.std(std_speed))
print("pos", np.average(std_pos), np.std(std_pos))
