from x_transformers import AutoregressiveWrapper, TransformerWrapper, Decoder
from models.transformers import EncoderGearFormer
import torch

def loading_model(args, input_size, output_size, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Loading the encoder and decoder
    input:
    ------
    input_size: this is the size of input which is 6 if you pass weight as input and 5 if we don't pass the weight
    output_size: this is the vocab-size (53 for our case)
    max_length: this is the maximum length of sequence in our dataset (21 including start and end)

    returns:
    ------
    decoder and encoder based on the model we want to use
    """
    encoder = EncoderGearFormer(input_size, args.dim , depth = args.depth).to(device)
    model_decoder = TransformerWrapper(
    num_tokens = output_size,
    max_seq_len = max_length - 1,    # max_length is 19 + 2 (start and end), here I use max_length - 1(start), since we are already passing start to transformer
    attn_layers = Decoder(
        dim = args.dim,
        depth = args.depth,
        heads = args.head,
        cross_attend = True
        )
    )

    decoder = AutoregressiveWrapper(model_decoder, ignore_index=output_size-1).to(device)

    
    return encoder, decoder