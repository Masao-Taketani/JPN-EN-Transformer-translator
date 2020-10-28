import sentencepiece as spm
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

from models.Transformer.layers import *
from utils import *


BUFFER_SIZE = 20000
BATCH_SIZE = 64
TR_TE_RATIO = 0.05
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

    

if __name__ == "__main__":
    main()
