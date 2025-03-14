import random
random.seed(0)
import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import os
import torch.optim as optim
import torch.nn as nn
from  utils.data_handle import load_data
from utils.config_file import config
from utils.helper import  get_coef
from models.load_model import loading_model
from models.transformers import train_gearformer, val_gearformer
from tqdm import tqdm
from evaluation import Eval
import csv

if __name__ == "__main__":
    
    args = config()    
    model = args.model_name
    get_data = load_data(args)
    weight_c = get_coef(args, get_data)
    train_loader, input_size = get_data.get_data_for_dl(args.BS, if_val=False, if_weight=args.if_weight)
    val_loader, _ = get_data.get_data_for_dl(args.BS, if_val=True, if_weight=args.if_weight)
    encoder, decoder = loading_model(args, input_size, get_data.output_size, get_data.max_length)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)


    total_loss_val, val_len = 0, 0
    total_loss_train, train_len = 0, 0
    adaptive_weight = torch.tensor(np.pi)/2
    cros_loss_train, cros_loss_val = 0, 0
    w_loss_train, w_loss_val = 0, 0
    val_vec = [100]
    for epoch in range(args.epoch):
        encoder.train()
        decoder.train()     
               
        for batch_idx, (x_train, y_train, target_length, weight) in enumerate(tqdm(train_loader)):
            loss , loss_cros, loss_w= train_gearformer(x_train, y_train, target_length, get_data.output_size, encoder, decoder, encoder_optimizer, decoder_optimizer, weight_c, adaptive_weight, loss_weight=args.WWL)
            cros_loss_train += loss_cros * len(x_train)
            w_loss_train += loss_w * len(x_train)
            total_loss_train += loss * len(x_train)
            train_len += len(x_train)

        for batch_idx, (x_val, y_val, target_length, weight_val) in enumerate(tqdm(val_loader)):
            loss , loss_cros, loss_w = val_gearformer(x_val, y_val, target_length, get_data.output_size, encoder, decoder, weight_c, adaptive_weight, loss_weight=args.WWL)
            total_loss_val += loss * len(x_val)
            w_loss_val += loss_w * len(x_val)
            cros_loss_val += loss_cros * len(x_val)
            val_len += len(x_val)


        print(epoch, "train Loss:", total_loss_train/train_len, "cross:", cros_loss_train/train_len, "weight", w_loss_train/train_len) 
        print(epoch, "Val Loss:", total_loss_val/val_len, "cross:", cros_loss_val/val_len, "weight", w_loss_val/val_len)
        adaptive_weight = max(torch.tensor(0), adaptive_weight-np.pi/6)

        if epoch == 0 or cros_loss_val/val_len < val_vec[-1]:
            val_vec.append(cros_loss_val/val_len)
            encoder_chackpoint_name = "GearFormer_" + str(args.lr) + "_" + str(epoch) + "_encoder.dict"
            decoder_chackpoint_name = "GearFormer_" + str(args.lr) + "_" + str(epoch) + "_decoder.dict"
            print("Saving model")
            best_model = total_loss_val/val_len
            torch.save(encoder.state_dict(), os.path.join(args.checkpoint_path, encoder_chackpoint_name))
            torch.save(decoder.state_dict(), os.path.join(args.checkpoint_path, decoder_chackpoint_name))

            csv_file_name = "GearFormer_EPOCH"+str(epoch)+"_BS"+str(args.BS)+"_WWL"+str(args.WWL)+"_lr"+str(args.lr)+"_gamma"+".csv"

            csvfile = open(os.path.join(args.checkpoint_path, csv_file_name), 'w')
            csvwriter = csv.writer(csvfile)
            eval = Eval(args, get_data.output_size, args.model_name, encoder, decoder)
            
            kwargs = {'num_workers': 0} if torch.cuda.is_available() else {}

            dataset_length = 0
            all_correct , all_valid, all_grammar = 0, 0, 0
            for batch_idx, (x_val, y_val, target_length, _) in enumerate(tqdm(val_loader)):
                correct , valid, grammar = eval.get_output_sequence_accuracy(x_val, y_val, csvwriter)
                all_correct += correct
                all_valid += valid
                all_grammar += grammar
                dataset_length += len(y_val)
            
            print(all_correct/dataset_length , all_valid/dataset_length, all_grammar/dataset_length)

        else:
            break
            


