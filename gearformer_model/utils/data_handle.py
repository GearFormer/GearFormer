import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import os
import pickle
import json
import math
import csv
import pandas as pd


class MyDataset(torch.utils.data.Dataset):
    """
    x_train: if we are using weight as input (if_weight==True), it's a vector of size six [ratio(1), position(3), rotation(1), sequence_weight(1)] otherwise its [ratio(1), position(3), rotation(1)] with size five.
    y_train: it the sequence corresponding to the information in x_train
    weight: it's the sequence_weight
    """
    def __init__(self, x_train, y_train, target_length, weight):
        self.x_train = x_train.values
        self.y_train = y_train.values
        self.weight = weight.values
        self.target_length = target_length.values
    def __getitem__ (self, index):
        return torch.Tensor(self.x_train[index]).cuda(), torch.Tensor(eval(self.y_train[index])).cuda(), torch.Tensor([self.target_length[index]]).cuda(), torch.Tensor([self.weight[index]]).cuda()

    def __len__(self):
        return len(self.weight)

class load_data:
    def __init__(self, args):
        self.args = args
        self.tokens, self.inxs = self.get_tokens()
        self.output_size = len(self.tokens)
        self.max_length = 21

    def name2inx(self, name):
        return self.tokens[name]
    
    def inx2name(self, inx):
        return self.inxs[inx]

    def json_reader(self, filename):
        with open(filename) as f:
            data=json.load(f)
        return data
    
    def get_tokens(self):
        """
        This function reads the grammar.json files and returns two dictionary:
        1. tokens: which maps the tokens we have to unique indices
        2. inxs: which maps indices to tokens
        """
        inxs = {0:"<start>"}
        tokens = {"<start>":0}
        language = self.json_reader(self.args.language_path)
        language_vocabs = language['vocab']
        cnt = 1
        for i in language_vocabs:
            for j in language_vocabs[i]:
                if j not in tokens:
                    tokens[j] = cnt
                    inxs[cnt] = j
                    cnt += 1
        tokens["<end>"] = cnt
        inxs[cnt] = "<end>"
        cnt += 1
        tokens["EOS"] = cnt
        inxs[cnt] = "EOS"
        return tokens, inxs

    def get_all_data(self, if_val=False, if_weight=True):
        """
        input:
        ------
        if_val: if True loads the val data and the output corresponds to val set, otherwise it will be train set
        if_weight: if True, weight would be used in input

        returns:
        ------
        x_train: if we are using weight as input (if_weight==True), it's a vector of size six [ratio(1), position(3), rotation(1), sequence_weight(1)] otherwise its [ratio(1), position(3), rotation(1)] with size five.
        y_train: it the sequence corresponding to the information in x_train
        weight: it's the sequence_weight
        max(weight): returns the weight of the sequence with maximum weight from the set
        """

        if if_val:
            folder = self.args.val_data_path
        else:
            folder = self.args.train_data_path

        df = pd.read_csv(folder)
        weight = df.iloc[:,0]
        target_length = df.iloc[:,1]
        x_train = df.iloc[:,2:-1]
        y_train = df.iloc[:,-1]

        return x_train, y_train, target_length, weight, 0


    def get_data_for_dl(self, BS, if_val, if_weight):
        """
        This function load the data and returns the dataloader - makes the data ready for training
        """
        x, y, target_length, weight, _ = self.get_all_data(if_val=if_val, if_weight=if_weight)
        kwargs = {'num_workers': 0} if torch.cuda.is_available() else {}
        loader = torch.utils.data.DataLoader(MyDataset(x, y, target_length, weight), batch_size= BS, shuffle=True, **kwargs)
        return loader, x.shape[1]



if __name__ == "__main__":

 from config_file import config
 args = config()
 ld = load_data(args)
 print(ld.tokens)
 print(ld.inxs)