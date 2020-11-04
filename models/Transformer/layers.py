import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding
from tensorflow.keras import Model


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array: 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array: 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  # add the new dim to the first dimension
  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


# Scaled Dot-Product Attention
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be boradcastable for addition.
  Padding is used for self-attention and encoder-decoder attention.
  Look ahead is used for encoder-decoder attention.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
      to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  # shape: (..., seq_len_q, seq_len_k)
  matmul_qk = tf.matmul(q, k, transpose_b=True)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  # As the softmax normalization is done on k, its values decide the amount of
  # importance given to Q.
  # shape: (..., seq_len_q, seq_len_k)
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
  # shape: (..., seq_len_q, depth_v)
  output = tf.matmul(attention_weights, v)

  return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([Dense(dff,
                                    activation="relu"), #(batch_size, seq_len, dff)
                              Dense(d_model)]) # (batch_size, seq_len, d_model)


class MultiHeadAttention(Layer):

  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = Dense(d_model)
    self.wk = Dense(d_model)
    self.wv = Dense(d_model)

    self.dense = Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is
    (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    # all shapes: (batch_size, seq_len, d_model)
    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    # all shapes: (batch_size, num_heads, seq_len, depth)
    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)
    v = self.split_heads(v, batch_size)

    # scaled_attention.shape == (batch_size, num_heads, seq_len, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    scaled_attention, attention_weights = scaled_dot_product_attention(q,
                                                                       k,
                                                                       v,
                                                                       mask)
    # shape: (batch_size, seq_len, num_heads, depth)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
    # shape: (batch_size, seq_len, d_model)
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))
    # shape: (batch_size, seq_len, d_model)
    output = self.dense(concat_attention)

    return output, attention_weights


class EncoderLayer(Layer):

  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    # point_wise_feed_forward_netowork is a function
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = LayerNormalization(epsilon=1e-6)
    self.layernorm2 = LayerNormalization(epsilon=1e-6)
    """
    tf.keras.layers.Dropout(rate):
    rate: Float between 0 and 1. Fraction of the input units to drop.
    The Dropout layer randomly sets input units to 0 with a frequency of rate
    at each step during training time, which helps prevent overfitting.
    Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over
    all inputs is unchanged."""
    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)

  def call(self, x, training, mask):
    # shape: (batch_size, input_seq_len, d_model)
    atten_output, _ = self.mha(x, x, x, mask)
    atten_output = self.dropout1(atten_output, training=training)
    # shape: (batch_size, input_seq_len, d_model)
    out1 = self.layernorm1(x + atten_output)

    # shape: (batch_size, input_seq_len, d_model)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    # shape: (batch_size, input_seq_len, d_model)
    out2 = self.layernorm2(out1 + ffn_output)

    return out2


class Encoder(Layer):

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               input_vocab_size,
               maximum_position_encoding,
               rate=0.1):

    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.embedding = Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.d_model)
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
    self.dropout = Dropout(rate)

  def call(self, x, training, mask):
    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    # shape: (batch_size, input_seq_len, d_model)
    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dropout(x, training=training)
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    # shape: (batch_size, input_seq_len, d_model)
    return x


class DecoderLayer(Layer):

  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = LayerNormalization(epsilon=1e-6)
    self.layernorm2 = LayerNormalization(epsilon=1e-6)
    self.layernorm3 = LayerNormalization(epsilon=1e-6)

    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)
    self.dropout3 = Dropout(rate)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    # shape: (batch_size, target_seq_len, d_model)
    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(x + attn1)
    # shape: (batch_size, target_seq_len, d_model)
    attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
    attn2 = self.dropout2(attn2, training=training)
    # shape: (batch_size, target_seq_len, d_model)
    out2 = self.layernorm2(out1 + attn2)

    # shape: (batch_size, target_seq_len, d_model)
    ffn_output = self.ffn(out2)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(out2 + ffn_output)

    return out3, attn_weights_block1, attn_weights_block2


class Decoder(Layer):

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               target_vocab_size,
               maximum_position_encoding,
               rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.embedding = Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                                    for _ in range(num_layers)]
    self.dropout = Dropout(rate)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    seq_len = tf.shape(x)[1]
    attention_weights = {}
    # shape: (batch_size, target_seq_len, d_model)
    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dropout(x, training=training)
    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x,
                                             enc_output,
                                             training,
                                             look_ahead_mask,
                                             padding_mask)

      attention_weights["decoder_layer{}_block1".format(i+1)] = block1
      attention_weights["decoder_layer{}_block2".format(i+1)] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


class Transformer(Model):

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               input_vocab_size,
               target_vocab_size,
               pe_input, pe_target,
               rate=0.1):
    super(Transformer, self).__init__()
    self.encoder = Encoder(num_layers,
                           d_model,
                           num_heads,
                           dff,
                           input_vocab_size,
                           pe_input,
                           rate)
    self.decoder = Decoder(num_layers,
                           d_model,
                           num_heads,
                           dff,
                           target_vocab_size,
                           pe_target,
                           rate)
    self.final_layer = Dense(target_vocab_size)

  def call(self,
           inp,
           tar,
           training,
           enc_padding_mask,
           look_ahead_mask,
           dec_padding_mask):
    # shape: (batch_size, inp_seq_len, d_model)
    enc_output = self.encoder(inp, training, enc_padding_mask)
    # shape: (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(tar,
                                                 enc_output,
                                                 training,
                                                 look_ahead_mask,
                                                 dec_padding_mask)
    # shape: (batch_size, tar_seq_len, target_vocab_size)
    final_output = self.final_layer(dec_output)

    return final_output, attention_weights
