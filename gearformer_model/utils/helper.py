import json
from decimal import Decimal as D
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch


def json_reader(filename):
    with open(filename) as f:
        data=json.load(f)
    return data

def get_coef(args, get_data):
    """
    input
    ----
    get_data: an instance of "load_data" class from data_handle.py

    return
    ----
    weight_c: a coeff vector which is going to be used in calculating the weight of each sequence
    """
    catalouge = json_reader(args.catalogue_path)
    coeff = {}
    max_ = 0

    for i in catalouge:
        coeff[i] = catalouge[i]["weight"]
        if coeff[i] > max_: max_ = coeff[i]
    

    weight_c = []
    for i in get_data.tokens:
        if i in coeff:
            weight_c.append(coeff[i]/max_)
        else:
            weight_c.append(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.Tensor(weight_c).to(device)

def concat_generators(*args):
      for gen in args:
          yield from gen

def is_grammatically_correct(args, seq):
    """
    input
    ----
    seq: a sequence of tokens

    return
    ----
    True: if this sequence respects the grammar
    False: if this sequence does not respect the grammar
    """
    language = json_reader(args.language_path)
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
    if len(seq) < 5:
        return False
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
    
def is_physically_feasible(seq, catalogue_path):
    """
    Given a sequence, we want to see if it is physically feasible or not

    input
    ----
    seq: a sequence of tokens
    args: uses the path to catalouge from args

    return
    ----
    True: if the seq is physically feasible
    False: if the seq is not physically feasible
    """
    catalogue = json_reader(catalogue_path)
    hypoid_sign = 1
    occupied = []
    ortho1 = {"x":"y", "y":"z", "z":"x"}
    ortho2 = {"x":"z", "y":"x", "z":"y"}
    dir_to_vec = {"x":np.array([1,0,0]), "y":np.array([0,1,0]), "z":np.array([0,0,1])}
    needed_space = 0
    curr = np.array([D(0),D(0),D(0)])
    i = 0
    curr_dir = "y"

    mhpl_translate = False

    while (i < len(seq)-1):
        i += 1          
        if "MSGA" in seq[i] or "SBSG" in seq[i] or "MMSG" in seq[i] or "AG" in seq[i] or "SWG" in seq[i] or ("MHP" in seq[i] and "R" in seq[i]):
            radius = D(catalogue[seq[i]]["outside_dia"] / 2)
            width = D(catalogue[seq[i]]["total_length"] / 2)
            bb = dir_to_vec[curr_dir]*width + dir_to_vec[ortho1[curr_dir]]*radius + dir_to_vec[ortho2[curr_dir]]*radius
            needed_space = ((curr[0]-bb[0], curr[0]+bb[0]), (curr[1]-bb[1], curr[1]+bb[1]), (curr[2]-bb[2], curr[2]+bb[2]))
            
        elif "MRGF" in seq[i]:
            width = D(catalogue[seq[i]]["face_width"]/2)
            height = D(catalogue[seq[i]]["height"] / 2)
            length = D(catalogue[seq[i]]["total_length"])
            if seq[i-1] == "<start>":
                curr_dir = "x"
                if "1st" in seq[i+1]:
                    pitch_side = "y"
                    width_side = "z"
                else:
                    pitch_side = "z"
                    width_side = "y"
                bb = dir_to_vec[curr_dir]*length + dir_to_vec[pitch_side]*height + dir_to_vec[width_side]*width
                needed_space = ((curr[0], curr[0]+2*length), (curr[1]-bb[1], curr[1]+bb[1]), (curr[2]-bb[2], curr[2]+bb[2]))

            elif seq[i+1] == "<end>":
                a = ["x", "y", "z"]
                a.remove(curr_dir)
                a.remove(pitch_side)
                if len(a) != 1:
                    Error
                else:
                    width_side = a[0]
                
                bb = dir_to_vec[curr_dir]*length + dir_to_vec[pitch_side]*height + dir_to_vec[width_side]*width
                needed_space = ((curr[0]-bb[0], curr[0]+bb[0]), (curr[1]-bb[1], curr[1]+bb[1]), (curr[2]-bb[2], curr[2]+bb[2]))


        elif "MHP" in seq[i] and "L" in seq[i]:
            length = D(catalogue[seq[i]]["total_length"])
            radius = D(catalogue[seq[i]]["shaft_dia"] / 2)
            if curr_dir == "x":
                if hypoid_sign == 1:
                    needed_space = ((curr[0], curr[0]+length), (curr[1]-radius, curr[1]+radius), (curr[2]-radius, curr[2]+radius))
                else:
                    needed_space = ((curr[0]-length, curr[0]), (curr[1]-radius, curr[1]+radius), (curr[2]-radius, curr[2]+radius))

            if curr_dir == "y":
                if hypoid_sign == 1:
                    needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1], curr[1]+length), (curr[2]-radius, curr[2]+radius))
                else:
                    needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1]-length, curr[1]), (curr[2]-radius, curr[2]+radius))

            if curr_dir == "z":
                if hypoid_sign == 1:
                    needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1]-radius, curr[1]+radius), (curr[2], curr[2]+length))
                else:
                    needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1]-radius, curr[1]+radius), (curr[2]-length, curr[2]))      


        elif "mesh" in seq[i]:
            if "plus" in seq[i]: sign_ = 1
            else: sign_ = -1
            
            if "MSGA" in seq[i+1] and "MSGA" in seq[i-1]:   ### spur
                r1 = D(catalogue[seq[i-1]]["pitch_dia"])/2
                r2 = D(catalogue[seq[i+1]]["pitch_dia"])/2

                if "1st" in seq[i]: 
                    curr += dir_to_vec[ortho1[curr_dir]] * sign_ * (r1+r2)

                else:
                    curr += dir_to_vec[ortho2[curr_dir]] * sign_ * (r1+r2)



            elif "MHP" in seq[i+1] and "MHP" in seq[i-1]:   ### bevel hypoid and pinion
                if "R" in seq[i-1]:   ### i+1 is hypoid and i-1 is bevel
                    hypoid_sign *= -1
                    e1 = D(catalogue[seq[i-1]]["mounting_distance"])
                    i1 = D(catalogue[seq[i-1]]["total_length"])/2
                    e2 = D(catalogue[seq[i+1]]["mounting_distance"])
                    l = D(catalogue[seq[i-1]]['offset'])

                    if "1st" in seq[i]:
                        curr += dir_to_vec[curr_dir]*(e1-i1)*t_sign + dir_to_vec[ortho1[curr_dir]]*(sign_*l) + dir_to_vec[ortho2[curr_dir]]*(sign_*e2)
                        curr_dir = ortho2[curr_dir]

                    elif "2nd" in seq[i]:
                        curr += dir_to_vec[curr_dir]*(e1-i1)*t_sign + dir_to_vec[ortho2[curr_dir]]*(sign_*l) + dir_to_vec[ortho1[curr_dir]]*(-1*sign_*e2)
                        curr_dir = ortho1[curr_dir]

                
                if "L" in seq[i-1]: ### i+1 is bevel and i-1 is hypoid
                    e1 = D(catalogue[seq[i-1]]["mounting_distance"])
                    i2 = D(catalogue[seq[i+1]]["total_length"])/2
                    e2 = D(catalogue[seq[i+1]]["mounting_distance"])
                    l = D(catalogue[seq[i-1]]['offset'])

                    if "1st" in seq[i]:
                        curr += dir_to_vec[curr_dir]*e1*t_sign + dir_to_vec[ortho1[curr_dir]]*(-1*sign_*l) + dir_to_vec[ortho2[curr_dir]]*(-1*sign_*(e2-i2))
                        curr_dir = ortho2[curr_dir]
                    
                    elif "2nd" in seq[i]:
                        curr += dir_to_vec[curr_dir]*e1*t_sign + dir_to_vec[ortho1[curr_dir]]*(-1*sign_*(e2-i2)) + dir_to_vec[ortho2[curr_dir]]*(sign_*l)
                        curr_dir = ortho1[curr_dir]
                    

            elif ("SBSG" in seq[i+1] and "SBSG" in seq[i-1]) or ("MMSG" in seq[i+1] and "MMSG" in seq[i-1]):   ### bevel or meter
                    
                e1 = D(catalogue[seq[i-1]]["mounting_distance"])
                e2 = D(catalogue[seq[i+1]]["mounting_distance"]) 
                i1 = D(catalogue[seq[i-1]]["total_length"]/2)
                i2 = D(catalogue[seq[i+1]]["total_length"]/2)
                
                if "1st" in seq[i]:
                    curr += dir_to_vec[curr_dir]*(e1-i1)*t_sign + dir_to_vec[ortho1[curr_dir]]*sign_*(e2-i2)
                    curr_dir = ortho1[curr_dir]

                if "2nd" in seq[i]:
                    curr += dir_to_vec[curr_dir]*(e1-i1)*t_sign + dir_to_vec[ortho2[curr_dir]]*sign_*(e1-i1)
                    curr_dir = ortho2[curr_dir]
                    

            elif "SWG" in seq[i-1] or "SWG" in seq[i+1]:
                r1 = D(catalogue[seq[i-1]]["pitch_dia"])/2
                r2 = D(catalogue[seq[i+1]]["pitch_dia"])/2
                if "1st" in seq[i]:
                    curr += dir_to_vec[ortho1[curr_dir]]*sign_*(r1+r2)
                    curr_dir = ortho2[curr_dir]

                elif "2nd" in seq[i]:
                    curr += dir_to_vec[ortho2[curr_dir]]*sign_*(r1+r2)
                    curr_dir = ortho1[curr_dir]


            elif "MRGF" in seq[i-1] or "MRGF" in seq[i+1]:
                if "MRGF" in seq[i-1]:
                    total_length = D(catalogue[seq[i-1]]["total_length"])
                    height = D(catalogue[seq[i-1]]["height_to_pitch"])/2    ### chose height(high_to_pich_line) for rack and pitch_dia(out_side_dia) for gear
                    pitch_dia = D(catalogue[seq[i+1]]["pitch_dia"])/2
                    if "1st" in seq[i]:
                        curr += dir_to_vec[curr_dir]*total_length + dir_to_vec[ortho1[curr_dir]]*sign_*(height+pitch_dia)
                        curr_dir = ortho2[curr_dir]

                    elif "2nd" in seq[i]:
                        curr += dir_to_vec[curr_dir]*total_length + dir_to_vec[ortho2[curr_dir]]*sign_*(height+pitch_dia)
                        curr_dir = ortho1[curr_dir]

                elif "MRGF" in seq[i+1]: ### this is the end we do not need the curr anymore
                    total_length = D(catalogue[seq[i+1]]["total_length"])
                    height = D(catalogue[seq[i+1]]["height_to_pitch"])/2    ### chose height(high_to_pich_line) for rack and pitch_dia(out_side_dia) for gear
                    pitch_dia = D(catalogue[seq[i-1]]["pitch_dia"])/2
                    if "1st" in seq[i]:
                        curr += dir_to_vec[ortho1[curr_dir]]*sign_*(height+pitch_dia)
                        curr_dir = ortho2[curr_dir]
                        pitch_side = ortho1[curr_dir]

                    elif "2nd" in seq[i]:
                        curr += dir_to_vec[ortho2[curr_dir]]*sign_*(height+pitch_dia)
                        curr_dir = ortho1[curr_dir]
                        pitch_side = ortho2[curr_dir]
            

        elif "translate" in seq[i]:
            if ("MHP" in seq[i-1]) and ("L" in seq[i-1]):
                mhpl_translate = True

            if "plus" in seq[i]: t_sign = +1
            else: t_sign = -1

            if seq[i+1] == "SH-*":
                length = 0
                if seq[i-1] == "<start>" or ("MHP" in seq[i-1] and "R" in seq[i-1]):
                    pass
                else:                        
                    length += D(catalogue[seq[i-1]]["total_length"] / 2)
                
                if seq[i+2] == "<end>" or ("MHP" in seq[i-1] and "R" in seq[i-1]):
                    pass
                else:
                    length += D(catalogue[seq[i+2]]["total_length"] / 2)

                
                if curr_dir == "y":
                    curr = (curr[0], curr[1]+t_sign*length, curr[2])
                if curr_dir == "x":
                    curr = (curr[0]+t_sign*length, curr[1], curr[2])
                if curr_dir == "z":
                    curr = (curr[0], curr[1], curr[2]+t_sign*length)


            else:
                length = D(catalogue[seq[i+1]]["total_length"])
                radius = D(catalogue[seq[i+1]]["outside_dia"]/2)

                if "plus" in seq[i]:
                    hypoid_sign = +1
                    if curr_dir == "y":
                        needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1], curr[1]+length), (curr[2]-radius, curr[2]+radius))
                        curr = (curr[0], curr[1]+length, curr[2])
                    if curr_dir == "x":
                        needed_space = ((curr[0], curr[0]+length), (curr[1]-radius, curr[1]+radius), (curr[2]-radius, curr[2]+radius))
                        curr = (curr[0]+length, curr[1], curr[2])
                    if curr_dir == "z":
                        needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1]-radius, curr[1]+radius), (curr[2], curr[2]+length))
                        curr = (curr[0], curr[1], curr[2]+length)
                else:
                    hypoid_sign = -1
                    if curr_dir == "y":
                        needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1]-length, curr[1]), (curr[2]-radius, curr[2]+radius))
                        curr = (curr[0], curr[1]-length, curr[2])
                    if curr_dir == "x":
                        needed_space = ((curr[0]-length, curr[0]), (curr[1]-radius, curr[1]+radius), (curr[2]-radius, curr[2]+radius))
                        curr = (curr[0]-length, curr[1], curr[2])
                    if curr_dir == "z":
                        needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1]-radius, curr[1]+radius), (curr[2]-length, curr[2]))
                        curr = (curr[0], curr[1], curr[2]-length)

                
                i += 1
        if needed_space != 0:
            if possible(occupied, needed_space, mhpl_translate):
                occupied.append(needed_space)
                mhpl_translate = False
                needed_space = 0
            else:
                return False
    return True            

def possible(occupied, needed_space, mhpl_translate):
    if not mhpl_translate:
        occupied = occupied[:-1]  
    for i in occupied:
        if intervals_collide(i[1], needed_space[1]):
            if intervals_collide(i[0], needed_space[0]):
                if intervals_collide(i[2], needed_space[2]):
                    return False
    return True
            
    
def intervals_collide(interval1, interval2):
    if interval2[0] == interval1[0] and interval2[1] == interval1[1]:
        return True
    for point in interval1:
        if (interval2[0] < point < interval2[1]):
            return True
    for point in interval2:
        if (interval1[0] < point < interval1[1]):
            return True
    return False


class generate_sequences:

    def __init__(self, args):
        self.args = args
        if self.args.max_length < 2:
            print("The maximum number of components should be at least 2")
            return 0
        # self.args.max_length += 2 ### plus start and end

    def get_number_of_components(self, sequence):
        """
        Given a sequence computes how many components it has, 
        IMPORTANT: because we added start and end to the number of componenets we 
        only remove mesh and Translate to compute the number of components.
        """

        elements_to_remove = {"Mesh", "Translate", "<start>", "<end>"}
        return len([word for word in sequence if word not in elements_to_remove])

    def get_policy(self):
        self.catalogue = json_reader(self.args.catalogue_path)
        language = json_reader(self.args.language_path)
        grammar_list = language['grammar']
        self.grammar = {}
        for elem in grammar_list:
            self.grammar[elem['LHS']] = [sublist for sublist in elem['RHS']]
        self.vocab = language['vocab']

    def add_to_seq(self, current_sequence):
        """
        Given an incoplete sequence format of length L,
        it returns all the valid sequence formats of length L+1
        """
        possible_seqs = []    

        for rhs in self.grammar[current_sequence[-1]]:


            possible_seqs.append(current_sequence + rhs)
        return possible_seqs

    def generate_seq_format(self):
        """
        Generates all the possible sequence formats
        using the grammer in the language.json
        """
        self.get_policy()
        
        completed_seq = set()
        processing_seqs = [["<start>"]]

        while len(processing_seqs) > 0 :

            for elem in processing_seqs:
                if elem[-1] == "<end>":
                    if self.get_number_of_components(elem) > 1 and (self.get_number_of_components(elem) < self.args.max_length + 1):
                        completed_seq.add(tuple(elem))
                else:
                    if self.get_number_of_components(elem) < self.args.max_length + 1:
                        processing_seqs += self.add_to_seq(elem)
                        
                processing_seqs.remove(elem)

        return completed_seq
    

    def generate_sequence_random(self, seq_format):
        self.get_policy()
        seq_format = seq_format[1:-1]
        seq = []
        for i in seq_format:
            seq.append(random.choice(self.vocab[i]))
        return seq







