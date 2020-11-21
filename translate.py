import os
import argparse
#import warnings
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utils.utils import *
from models.Transformer.layers import Transformer


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("jpn_input", help="input Japanese text.")
args = parser.parse_args()

num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
jpn_vocab_size = 8000
en_vocab_size = 8000
ckpt_path = "models/Transformer/ckpt/"
JPN_MAX_LEN = 100
EN_MAX_LEN = 100


def main():
    transformer = Transformer(num_layers=num_layers,
                              d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,
                              input_vocab_size=jpn_vocab_size,
                              target_vocab_size=en_vocab_size,
                              pe_input=JPN_MAX_LEN,
                              pe_target=EN_MAX_LEN)

    learning_rate = CustomSchedule(d_model)
    optimizer = Adam(learning_rate,
                     beta_1=0.9,
                     beta_2=0.98,
                     epsilon=1e-9)

    latest = tf.train.latest_checkpoint(ckpt_path)
    #latest = os.path.join(ckpt_path, "ckpt-6")
    checkpoint = tf.train.Checkpoint(transformer=transformer,
                                     optimizer=optimizer)
    checkpoint.restore(latest)

    translate(args.jpn_input, EN_MAX_LEN, checkpoint.transformer, result_log=True, plot="")


if __name__ == "__main__":
    main()
