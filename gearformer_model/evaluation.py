import torch
from utils.data_handle import load_data
from decimal import Decimal as D
from decimal import getcontext
import csv
from utils.helper import is_grammatically_correct, is_physically_feasible
from  utils.data_handle import load_data, MyDataset
from tqdm import tqdm
from models.load_model import loading_model
import os
getcontext().prec = 5
from utils.config_file import config
import random
random.seed(0)
import numpy as np
np.random.seed(0)
torch.manual_seed(0)


class Eval(load_data):
    def __init__(self, args, output_size, model, encoder, decoder):
        super(Eval, self).__init__(args)
        self.args = args
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.encoder.cuda().eval()
        self.decoder.cuda().eval()
        self.output_size = output_size

    def get_output_sequence_accuracy(self, x_val, y_val, csvwriter):
        correct = 0
        valid = 0
        grammar = 0
        
        input_vec = x_val.clone().detach().to(torch.float32).cuda()
        decoder_input = torch.zeros(self.output_size).cuda()
        decoder_input[0] = 1
        decoder_input = decoder_input.repeat(input_vec.shape[0], 1).cuda()

        with torch.no_grad():
                
            encoded_input = self.encoder(input_vec)
            prompt = torch.zeros((len(y_val),1)).cuda()
            all_out = self.decoder.generate(prompts=prompt, context=encoded_input, seq_len=20)

        for inx in range(len(all_out)):
            out = all_out[inx]

            out = list(map(self.inx2name, out.cpu().tolist()))

            out.append("<end>")
            target_inx = out.index("<end>")
            out = out[:target_inx+1]


            csvwriter.writerow([input_vec[inx][0],input_vec[inx][1], input_vec[inx][2], input_vec[inx][3], input_vec[inx][4], input_vec[inx][5], input_vec[inx][6], input_vec[inx][7], out])


            if is_grammatically_correct(self.args, ['<start>'] + out):
                grammar += 1
                if is_physically_feasible(['<start>'] + out, self.args.catalogue_path):
                    valid += 1
        return correct , valid, grammar


    def accuracy(self, seq1, seq2):
        seq2 = seq2[1:]
        for i in range(len(seq1)):
            if int(seq1[i]) != int(seq2[i]):
                return False
            if seq1[i] == 27:
                break
        return True

        
if __name__ == "__main__":
    args = config()
    max_length = 21
    output_size = 53 # number of classes
    with_weight = False
    input_size = 8

    csv_file_name = str(args.model_name)+"_EPOCH"+str(args.epoch)+"_BS"+str(args.BS)+"_WWL"+str(args.WWL)+"_lr"+str(args.lr)+".csv"

    csvfile = open(csv_file_name, 'w')
    csvwriter = csv.writer(csvfile)
    encoder, decoder = loading_model(args, input_size, output_size, max_length)
    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.encoder_chackpoint_name)))
    decoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.decoder_chackpoint_name)))

    ### This is to calculate the number of parameters:
    pytorch_total_params = sum(p.numel() for p in encoder.parameters())
    pytorch_total_params_t = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print("encoder:", pytorch_total_params, pytorch_total_params_t)
    pytorch_total_params = sum(p.numel() for p in decoder.parameters())
    pytorch_total_params_t = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print("decoder:", pytorch_total_params, pytorch_total_params_t)
    
    eval = Eval(args, output_size, args.model_name, encoder, decoder)

    x_val, y_val, target_length, weight_val, _ = eval.get_all_data(True, with_weight)


    dataset_length = len(y_val)

    kwargs = {'num_workers': 0} if torch.cuda.is_available() else {}
    print("dataset_length:  ", dataset_length )

    val_loader = torch.utils.data.DataLoader(MyDataset(x_val, y_val, target_length, weight_val), batch_size=args.BS, shuffle=False, **kwargs)
    all_correct , all_valid, all_grammar = 0, 0, 0
    for batch_idx, (x_val, y_val, target_length, _) in enumerate(tqdm(val_loader)):
        correct , valid, grammar = eval.get_output_sequence_accuracy(x_val, y_val, csvwriter)
        all_correct += correct
        all_valid += valid
        all_grammar += grammar
    
    print(all_correct/dataset_length , all_valid/dataset_length, all_grammar/dataset_length)
