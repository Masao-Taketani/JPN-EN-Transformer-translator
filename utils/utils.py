import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


jpn_sp_model = "models/spm/jpn_spm.model"
en_sp_model = "models/spm/en_spm.model"
jpn_sp = spm.SentencePieceProcessor()
en_sp = spm.SentencePieceProcessor()
jpn_sp.Load(jpn_sp_model)
en_sp.Load(en_sp_model)


def read_data(fpath):
    with open(fpath, "r") as f:
        return f.read()


def get_data(fpath):
    data = read_data(fpath)
    data_list = []
    for line in data.split("\n"):
        data_list.append(line)
    return data_list


def get_max_len_and_list(fpath):
    data = read_data(fpath)
    max_len = 0
    li = []
    for line in data.split("\n"):
        li.append(line)
        if max_len < len(line):
            max_len = len(line)
    return max_len, li


def get_max_len(data_list):
    max_len = 0
    for d in data_list:
        data_length = len(d)
        if data_length > max_len:
            max_len = data_length
    return max_len


# Masking
# The mask indicates where pad value 0 is present, it outputs a 1 at thoese
# locations. Otherwise 0.
def create_padding_mask(seq):
    # tf.math.equal(x, y): Returns the truth value of (x == y) element-wise.
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits
    # returned shape: (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(seq_len):
    """
    tf.linalg.band_part(input, num_lower, num_upper)
    ([used to be]tf.matrix_band_part)
    (e.g) tf.linalg.band_part(tf.ones((3, 3)), -1, 0)
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[1., 0., 0.],
         [1., 1., 0.],
         [1., 1., 1.]], dtype=float32)>
    [reference]https://dev.classmethod.jp/articles/tensorflow-matrixbandpart/
    """
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    # returned shape: (seq_len, seq_len)
    return mask


def create_masks(inp, tar):
    # Encode padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # this padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    """tf.(math.)maximumx: it returns element-wise maximum
     >>> x= tf.constant([1, 1, 1, 0])
     >>> y = tf.constant([1, 0, 0, 0])
     >>> tf.math.maximum(x, y)
     <tf.Tensor: shape=(4,), dtype=int32, numpy=array([1, 1, 1, 0], dtype=int32)>
    """
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# When you have tf.Tensor(string) and .numpy() method is used inside of the tf.py_function,
# it is converted to just a string. Not a numpy.
def encode(jpn, en):
    jpn_enc = [jpn_sp.PieceToId("<s>")] + jpn_sp.EncodeAsIds(jpn.numpy()) + [jpn_sp.PieceToId("</s>")]
    en_enc = [en_sp.PieceToId("<s>")] + en_sp.EncodeAsIds(en.numpy()) + [en_sp.PieceToId("</s>")]
    return jpn_enc, en_enc


def tf_encode(jpn, en):
    result_jpn, result_en = tf.py_function(encode, [jpn, en], [tf.int64, tf.int64])
    result_jpn.set_shape([None])
    result_en.set_shape([None])
    return result_jpn, result_en


def evaluate(inp_sentence, en_max_len, model):
    # inp sentence is Japanese, hence adding Japanese start and end token
    jpn_start_token = [jpn_sp.PieceToId("<s>")]
    jpn_end_token = [jpn_sp.PieceToId("</s>")]
    inp_sentence = jpn_start_token + jpn_sp.EncodeAsIds(inp_sentence) + jpn_end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is English, the first word to the transformer should be the
    # English start token
    decoder_input = [en_sp.PieceToId("<s>")]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(en_max_len):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input,
                                                                         output)

        # predictions.shape: (batch_size, seq_len, vocab_size)
        predictions, attention_weights = model(encoder_input,
                                               output,
                                               False,
                                               enc_padding_mask,
                                               combined_mask,
                                               dec_padding_mask)

    # select the last word from the seq_len dimension
    # shape: (batch_size, 1, vocab_size)
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if predicted_id == en_sp.PieceToId("</s>"):
          # tf.squeeze: Removes dimensions of size 1 from the shape of a tensor.
          #   axis: If specified, only squeezes the dimensions listed.
          return tf.squeeze(output, axis=0), attention_weights

    # concatenate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))
    sentence = jpn_sp.EncodeAsIds(sentence)
    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap="viridis")
        fontdict = {"fontsize": 10}
        ax.set_xticks(range(len(sentence)+2))
        ax.set_yticks(range(len(result)))
        ax.set_ylim(len(result)-1.5, -0.5)
        ax.set_xticklabels(["<s>"] + [jpn_sp.decode([i]) for i in sentence] + ["</s>"],
                           fontdict=fontdict, rotation=90)
        ax.set_yticklabels([en_sp.decode([i]) for i in result if i != en_sp.PieceToId("</s>")],
                           fontdict=fontdict)
        ax.set_xlabel("Head {}".format(head+1))

    plt.tight_layout()
    plt.show()


def translate(sentence, plot=""):
    result, attention_weights = evaluate(sentence)
    predicted_sentence = en_sp.decode([i for i in result if i != en_sp.PieceToId("</s>")])

    print("Input: {}".format(sentence))
    print("Predicted translation: {}".format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)


class CustomSchedule(LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
