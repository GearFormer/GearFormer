import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_path', type=str, default="utils/language.json", help='path to the language.json')
    parser.add_argument('--catalogue_path', type=str, default="utils/catalogue.json", help='path to the catalogue.json')
    parser.add_argument('--checkpoint_path', type=str, default="/app/gearformer_model/models", help = "path to the checkpoint folder")
    parser.add_argument('--train_data_path', type=str, default="/app/dataset/train_data.csv", help = "path to the train data folder")
    parser.add_argument('--val_data_path', type=str, default="/app/dataset/val_data.csv", help = "path to the val or test data folder")
    parser.add_argument('--epoch', type=int, default=19, help = "number of epochs")
    parser.add_argument('--BS', type=int, default=1024, help = "specify the batch size")
    parser.add_argument('--dim', type=int, default=512, help = "specify the dimention for gearformer")
    parser.add_argument('--depth', type=int, default=6, help = "specify the depth for gearformer")
    parser.add_argument('--head', type=int, default=8, help = "specify the number of head in gearformer decoder")
    parser.add_argument('--WWL', type=float, default=1, help = "specify the weight for sequence's weight in the loss")
    parser.add_argument('--model_name', type=str, default="GearFormer", help = "It should be GearFormer")
    parser.add_argument('--lr', type=float, default=0.0001 , help = "learning rat for optimizer")
    parser.add_argument('--if_weight', type=bool, default=False, help = "pass weight as input to the model if true - it's always false in current version")
    parser.add_argument('--encoder_chackpoint_name', type=str, default="GearFormer_0.0001_19_encoder.dict" , help = "name of the encoder checkpoint")
    parser.add_argument('--decoder_chackpoint_name', type=str, default="GearFormer_0.0001_19_decoder.dict" , help = "name of the decoder checkpoint")
    args = parser.parse_args()
    return args
