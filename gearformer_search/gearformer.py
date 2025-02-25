import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
from gearformer_model.utils.data_handle import load_data
import os
from gearformer_model.utils.config_file import config
from gearformer_model.models.load_model import loading_model

device = torch.device("cpu")
def gearformer(input_, model, args, max_length, seq):
    get_dict = load_data(args)

    input__size = len(input_)

    encoder, decoder = loading_model(model, input__size, get_dict.output_size, max_length)
    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.encoder_chackpoint_name), map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.decoder_chackpoint_name), map_location=torch.device('cpu')))
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_ = torch.tensor(input_).unsqueeze(0).to(torch.float32).to(device)
        encoded_input_ = encoder(input_)

        prompt = torch.zeros(len(seq))
        for i in range(len(seq)):
            prompt[i] = get_dict.name2inx(seq[i])
        prompt = prompt.to(device)
        loss_, (logits, _) = decoder(torch.zeros((1,21)).to(device).long(), context = encoded_input_, return_outputs = True)
        out = decoder.generate(prompts=prompt.unsqueeze(0), context=encoded_input_, seq_len=20-len(seq))
        out = list(map(get_dict.inx2name, out[0].cpu().tolist()))

    return out, logits

def autocomplete(input_, seq):
    args = config()
    max_length = 21

    # weight = 0.1
    out, logits = gearformer(input_, "Xtransformer", args, max_length, seq)
    new_seq = split_list(seq + out, "<end>")[0] + ["<end>"]

    return new_seq

def split_list(input_list, delimiter):
    result = []
    current_sublist = []

    for item in input_list:
        if item == delimiter:
            result.append(current_sublist)
            current_sublist = []
        else:
            current_sublist.append(item)

    # Add the last sublist if not empty
    if current_sublist:
        result.append(current_sublist)

    return result

