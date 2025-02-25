import sys
import torch
import random
random.seed(0)
import numpy as np
np.random.seed(0)
torch.manual_seed(0)
import torch.nn as nn
from einops import rearrange
from x_transformers import Encoder


class EncoderXtransformer(nn.Module):
    def __init__(self, input_size, output_size, depth):
        super(EncoderXtransformer, self).__init__()
        self.input_size = input_size
        self.bn = torch.nn.BatchNorm1d(input_size)
        self.output_size = output_size
        self.l1 = nn.Linear(input_size, output_size//2)
        self.bn1 = torch.nn.BatchNorm1d(output_size//2)
        self.l2 = nn.Linear(output_size//2, output_size)
        self.bn2 = torch.nn.BatchNorm1d(output_size)
        self.encoder = Encoder(dim = output_size, depth = depth).cuda()


    def forward(self, input):
        tensor1 = self.l1(input)
        tensor1 = nn.functional.relu(tensor1)
        tensor1 = self.bn1(tensor1)
        tensor2 = self.l2(tensor1)
        tensor2 = nn.functional.relu(tensor2)
        out = self.bn2(tensor2)
        out = self.encoder(out.unsqueeze(1))
        return out  
    

def train_xtransformer(input_data, target_tensor, target_length_seq, output_size, encoder, decoder, encoder_optimizer, decoder_optimizer, weight_c, adaptive_weight, loss_weight):
    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    ignore_index=output_size-1
    

    encoded_input = encoder(input_data)

    _, (logits, _) = decoder(target_tensor.long(), context = encoded_input, return_outputs = True) 


    target = target_tensor.long()[:, 1:]
    out_ = torch.nn.functional.gumbel_softmax(logits, dim=-1, hard=True)
    loss = torch.nn.functional.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index,
            reduction = "none"
    )

    weight = torch.matmul(out_, weight_c)
    loss = torch.div(loss.sum(dim=1).squeeze() , target_length_seq)
    loss_cros = loss
    loss_w = loss_weight*torch.cos(adaptive_weight)*weight.squeeze().mean(dim=1)
    loss = loss + loss_weight*torch.cos(adaptive_weight)*weight.squeeze().mean(dim=1)

    loss = loss.mean() 
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item(), loss_cros.mean().item(), loss_w.mean().item()


def val_xtransformer(input_data, target_tensor, target_length_seq, output_size, encoder, decoder, weight_c, adaptive_weight, loss_weight):
    encoder.eval()
    decoder.eval()
    ignore_index=output_size-1

    with torch.no_grad():
        encoded_input = encoder(input_data)
        _ , (logits, _) = decoder(target_tensor.long(), context = encoded_input, return_outputs = True) 
        target = target_tensor.long()[:, 1:]
        out_ = torch.nn.functional.gumbel_softmax(logits, dim=1, hard=True)

        loss = torch.nn.functional.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index,
            reduction = "none"
        )

        weight = torch.matmul(out_, weight_c)
        loss = torch.div(loss.sum(dim=1).squeeze() , target_length_seq)
        loss_cros = loss
        loss_w = loss_weight*torch.cos(adaptive_weight)*weight.squeeze().mean(dim=1)
        loss = loss + loss_weight*torch.cos(adaptive_weight)*weight.squeeze().mean(dim=1)
        loss = loss.mean() 


        return loss.item(), loss_cros.mean().item(), loss_w.mean().item()




    



# enc = Encoder(dim = 512, depth = 6)
# encoded_neighbors = enc(neighbors, mask = neighbor_masks)

# model_decoder = TransformerWrapper(
#     num_tokens = 20000,
#     max_seq_len = 1024,
#     attn_layers = Decoder(
#         dim = 512,
#         depth = 6,
#         heads = 8,
#         cross_attend = True
#     )
# )

# decoder = AutoregressiveWrapper(model_decoder)
# loss = decoder(target, context = encoded_neighbors) # (1, 1024, 20000)

# loss.backward

