import os
import sentencepiece as spm
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

from models.Transformer.layers import *
from utils.utils import *
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

jpn_txt_path = "dataset/jpn_data.txt"
en_txt_path = "dataset/en_data.txt"
ckpt_path = "models/Transformer/ckpt/"
log_path = "logs/"
EPOCHS = 45

def main():
    # load data from the data files
    jpn_data = get_data(jpn_txt_path)
    en_data = get_data(en_txt_path)
    train_jpn, val_jpn, train_en, val_en = train_test_split(jpn_data,
                                                            en_data,
                                                            test_size=TR_TE_RATIO)
    JPN_MAX_LEN = get_max_len(train_jpn)
    EN_MAX_LEN = get_max_len(train_en)
    # include [BOS] and [EOS] to each max len above
    JPN_MAX_LEN += 2
    EN_MAX_LEN += 2

    test_jpn_data = ["今日は夜ごはん何にしようかな？",
                     "ここ最近暑い日がずっと続きますね。",
                     "来年は本当にオリンピックが開催されるでしょうか？",
                     "将来の夢はエンジニアになることです。",
                     "子供のころはあの公園でたくさん遊んだなー。"]

    # preprocess for the train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_jpn, train_en))
    train_dataset = train_dataset.map(tf_encode)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(len(train_jpn)).padded_batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(BATCH_SIZE)
    ## preprocess for the validation dataset
    #val_dataset = tf.data.Dataset.from_tensor_slices((val_jpn, val_en))
    #val_dataset = val_dataset.map(tf_encode)
    #val_dataset = val_dataset.padded_batch(BATCH_SIZE)
    # preprocess for the test data
    test_dataset = tf.data.Dataset.from_tensor_slices((test_jpn_data))

    # instantiate the Transformer model
    transformer = Transformer(num_layers=num_layers,
                              d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,
                              input_vocab_size=jpn_vocab_size,
                              target_vocab_size=en_vocab_size,
                              pe_input=JPN_MAX_LEN,
                              pe_target=EN_MAX_LEN)
    # set learning rate, optimizer, loss and matrics
    learning_rate = CustomSchedule(d_model)
    optimizer = Adam(learning_rate,
                     beta_1=0.9,
                     beta_2=0.98,
                     epsilon=1e-9)

    loss_object = SparseCategoricalCrossentropy(from_logits=True,
                                                reduction="none")
    def loss_function(label, pred):
        mask = tf.math.logical_not(tf.math.equal(label, 0))
        loss_ = loss_object(label, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    train_loss = Mean(name="train_loss")
    train_accuracy = SparseCategoricalAccuracy(name="train_accuracy")

    """
    The @tf.function trace-compiles train_step into a TF graph for faster
    execution. The function specializes to the precise shape of the argument
    tensors. To avoid re-tracing due to the variable sequence lengths or
    variable batch sizes(usually the last batch is smaller), use input_signature
    to specify more generic shapes.
    """
    train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                            tf.TensorSpec(shape=(None, None), dtype=tf.int64)]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_label = tar[:, 1:]
        training = True

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp,
                                                                         tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp,
                                         tar_inp,
                                         training,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_label, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_label, predictions)


    # set the checkpoint and the checkpoint manager
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              ckpt_path,
                                              max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored.")

    # set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_path, current_time, "train")
    test_log_dir = os.path.join(log_path, current_time, "validation")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    #test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp: Japanese, tar: English
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 100 == 0:
                print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}"
                .format(epoch+1,
                        batch,
                        train_loss.result(),
                        train_accuracy.result()))

        # output the training log for every epoch
        print("Epoch {} Loss {:.4f} Accuracy {:.4f}".format(epoch+1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))
        print("Time taken for 1 epoch: {:.3f} secs\n".format(time.time() - start))

        # check how the model performs for every epoch
        test_summary_log = test_translate(test_jpn_data)

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch)
            tf.summary.text("test_text", test_summary_log)

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint for epoch {} at {}".format(epoch+1,
                                                                ckpt_save_path))


if __name__ == "__main__":
    main()
