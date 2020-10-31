import sentencepiece as spm
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

from models.Transformer.layers import *
from utils import *
from models.Transformer.layers import Transformer


BATCH_SIZE = 64
TR_TE_RATIO = 0.05

# hyperparameters
## base
num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
jpn_vocab_size = 8000
en_vocab_size = 8000

jpn_sp_model = "models/spm/jpn_spm.model"
en_sp_model = "models/spm/en_spm.model"
jpn_txt_path = "dataset/jpn_data.txt"
en_txt_path = "dataset/en_data.txt"
jpn_sp = spm.SentencePieceProcessor()
en_sp = spm.SentencePieceProcessor()
jpn_sp.Load(jpn_sp_model)
en_sp.Load(en_sp_model)


def main():
    # load data from the data files
    jpn_data = get_data(jpn_txt_path)
    en_data = get_data(en_txt_path)
    train_jpn, val_jpn, train_en, val_en = train_test_split(jpn_data,
                                                            en_data,
                                                            test_size=TR_TE_RATIO)
    JPN_MAX_LEN = get_max_len(train_jpn)
    EN_MAX_LEN = get_max_len(train_en)

    # preprocess for the train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_jpn, train_en))
    train_dataset = train_dataset.map(tf_encode)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(len(train_jpn)).padded_batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(BATCH_SIZE)
    # preprocess for the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_jpn, val_en))
    val_dataset = val_dataset.map(tf_encode)
    val_dataset = val_dataset.padded_batch(BATCH_SIZE)

    # instantiate the Transformer model
    transformer = Transformer(num_layers=num_layers,
                              d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,
                              input_vocab_size=jpn_vocab_size,
                              target_vocab_size=en_vocab_size,
                              pe_input=JPN_MAX_LEN,
                              pe_target=EN_MAX_LEN)



if __name__ == "__main__":
    main()
