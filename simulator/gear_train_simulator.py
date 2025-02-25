from gear_system import GearSystem
from util.parse_input import get_num_state_var
from util.compute_weight import compute_weight
import openmdao.api as om
import numpy as np

class Simulator:

    def run(self, input_data):
        num_state_var = get_num_state_var(input_data)

        prob = om.Problem(GearSystem(input_data = input_data), reports=None)
        prob.setup()

        prob.set_val('pos_0', [0, 0, 0])
        prob.set_val('flow_dir_0', [1, 0, 0])
        prob.set_val('motion_vector_0', [1, 0, 0])
        prob.set_val('q_dot_0', 1.0)

        prob.run_model()

        output_position = np.round(prob.get_val('pos_' + str(num_state_var - 1)), 9)
        output_rot_axis = np.round(prob.get_val('motion_vector_' + str(num_state_var - 1)), 9)
        output_speed = np.round(prob.get_val('q_dot_' + str(num_state_var - 1))[0], 9)

        weight = np.round(compute_weight(input_data["gear_train_sequence"]), 9)

        # except:
        #     output_position = [0, 0, 0]
        #     output_rot_axis = [0, 0, 0]
        #     output_speed = 0.0
        #     weight = 0.0
        if "MRGF" in input_data["gear_train_sequence"][1]:
            input_motion_type = "T"
        else:
            input_motion_type = "R"

        if "MRGF" in input_data["gear_train_sequence"][-2]:
            output_motion_type = "T"
        else:
            output_motion_type = "R"


        return {
            "id": input_data["id"],
            "input_motion_type": input_motion_type,
            "output_motion_type": output_motion_type,
            "gear_train_sequence" : input_data["gear_train_sequence"],
            "output_position": list(output_position),
            "output_motion_vector": list(output_rot_axis),
            "output_motion_speed": output_speed,
            "weight": weight
        }

if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description='Load a JSON file.')
    parser.add_argument('json_file', metavar='file_path', type=str, help='Path to the JSON file')
    args = parser.parse_args()
    with open(args.json_file, 'r') as file:
        input_data = json.load(file)
    simulator = Simulator()
    results = simulator.run(input_data)
    print(results)
