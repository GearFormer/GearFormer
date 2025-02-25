import json
def get_num_state_var(input_data):

    seq = input_data["gear_train_sequence"]
    num_state_var = 1
    for comp in seq:
        if "SH-" in comp:
            num_state_var += 1
        elif "mesh" in comp:
            num_state_var += 1

    return num_state_var

