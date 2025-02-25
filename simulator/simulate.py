from gear_train_simulator import Simulator
from concurrent.futures import ThreadPoolExecutor
from util.suppress_print import SuppressPrint
import os


import pickle
import argparse

simulator = Simulator()

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_cpus', type=int, default=4, help="how many cpus we want to use")
    parser.add_argument('--cpu_num', type=int, default=0, help="which cpu is")

    args = parser.parse_args()
    return args


def collect_pickle_contents(pickle_path):
  pickle_contents = {}
  id = 0
  with open(pickle_path, 'rb') as file:
    unpickler = pickle.Unpickler(file)
    while(True):
      try:
          seq = unpickler.load()
          # input_rot_axis = random.choice([1,-1])

          # input_speed = random.randint(1, 628)
          input_data = {
                        # "input_params": {
                        # "input_position": [0, 0, 0],
                        # "input_rot_axis": [0, input_rot_axis, 0],
                        # "input_speed": round(input_speed,2)
                        # },
                        "gear_train_sequence": list(['<start>'] + seq + ['<end>'])
                        }
          
          input_data["id"] = id
          
          with SuppressPrint():
            res = simulator.run(input_data)

          pickle_contents[id] = res
          id += 1
      except EOFError:
          break

  return pickle_contents

if __name__ == "__main__":
  args = config()   
  pickle_input_folder = '../dataset/examples/'
  pickle_output_folder = './simulator_output'
  cnt = args.cpu_num
  all_pkls = os.listdir(pickle_input_folder)
  while(cnt < len(all_pkls)):
    i = all_pkls[cnt]
    if not os.path.exists(os.path.join(pickle_output_folder, i)):
      print("Start", i)
      pickle_path = os.path.join(pickle_input_folder, i)
      json_data_list = collect_pickle_contents(pickle_path)
      with open(os.path.join(pickle_output_folder, i), 'wb') as pkl_out:
        pickle.dump(json_data_list, pkl_out)
      print("Finish", i)
    
    
    cnt += args.num_of_cpus