import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


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
