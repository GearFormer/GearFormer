
from utils.data_handle import load_data
from decimal import getcontext
getcontext().prec = 5
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
from utils.helper import generate_sequences, is_grammatically_correct, is_physically_feasible
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--language_path', type=str, default='utils/language.json', help='path to the language.json')
parser.add_argument('--max_length', type=int, default=10, help='maximum number of components - >= 2')
parser.add_argument('--catalogue_path', type=str, default='utils/catalogue.json', help='path to the catalogue.json')
parser.add_argument('--val_data_path', type=str, default="/home/ubuntu/data_folder_big/test_data.csv", help = "path to the val data folder")
args = parser.parse_args()

class Rand(load_data):
    def __init__(self, args):
        super(Rand, self).__init__(args)
        self.args = args
        self.seq_class = generate_sequences(args)
    def get_output_sequence_accuracy(self):
        input_vec, y_val, _, _, _ = self.get_all_data(True, False)
        input_vec = input_vec.values
        y_val_length = len(y_val)
        inx = 0
        valid = 0
        grammar = 0
        generated_format = self.seq_class.generate_seq_format()

        while (inx < y_val_length ):
            print(inx)
            all_seq = set()
            seq_format = random.choice(list(generated_format))
            seq_for_format = self.seq_class.generate_sequence_random(seq_format)
            if str(seq_for_format) not in all_seq: 
                all_seq.add(str(seq_for_format))
                csvwriter.writerow([input_vec[inx][0],input_vec[inx][1], input_vec[inx][2], input_vec[inx][3], input_vec[inx][4], input_vec[inx][5], input_vec[inx][6], input_vec[inx][7], seq_for_format])
                if is_grammatically_correct(self.args, ['<start>'] + seq_for_format + ['<end>']):
                    grammar += 1
                    if is_physically_feasible(['<start>'] + seq_for_format + ['<end>'], self.args.catalogue_path):
                        valid += 1

                else:
                    print("grammar wrong")
                    print(['<start>'] + seq_for_format + ['<end>'])
                    break
                
                inx += 1
        return valid/len(y_val), grammar/len(y_val)



if __name__ == "__main__":

    csvfile = open("random_baseline.csv", 'w')
    csvwriter = csv.writer(csvfile)
    random_baseline = Rand(args)

    accuracy = random_baseline.get_output_sequence_accuracy()
    print(accuracy)
    