# import string
# import pandas as pd
import re
import os
import numpy as np
import sentencepiece as spm
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
# import time
# from tqdm import tqdm
# from datetime import datetime
# from zoneinfo import ZoneInfo
# import shutil


#-----------------configuration values for model-------------------------
#configuration values for model
# BUFFER_SIZE = 20000  # Buffer size for shuffling data  #20000
# BATCH_SIZE = 32  # Batch size for training  #128
# EPOCHS = 60  # Number of training epochs   2 + 2 + 4 + 8 + 16 + 32 = 64  i.e., after 2, 4, 8, 16, 32, 64 epochs outputs
NUM_LAYERS = 4  # Transformer layers  #4
D_MODEL = 128  # Embedding dimension  #128
DFF = 512  # Feedforward layer size  #512
NUM_HEADS = 8  # Attention heads
SUMMARY_LENGTH = 16  # Max summary length  #16
TEXT_LENGTH = 512 # Max input text length   #512 (for muril: 512 max model sequence length)
# START_TOKEN = 1  # Start token for decoder
# END_TOKEN = 2  # End token for decoder
VOCAB_SIZE = 10000 #30000 #362191 #197285  # Total vocabulary size (for indic-bert: 200000, for muril: )
ENCODER_VOCAB_SIZE = VOCAB_SIZE  # Encoder vocabulary size
DECODER_VOCAB_SIZE = VOCAB_SIZE  # Decoder vocabulary size
VOCAB_DIM = 128  # Vocabulary embedding dimension
# CKPT_TO_KEEP = 5  # Number of checkpoints to keep
# SPLIT_RATIO = 0.80


#------------------------------Paths -----------------------
WEIGHTS_PATH = "./weights/Data 12.1"  #Epoch 21
# WEIGHTS_PATH = "./weights/Data 12"  #Epoch 20
TOKENIZER_DIR = "./tokenizer/Data 12"


#------------------ Positional Encoding---------------------------

def get_angles(position, i, d_model):
    base = 10000
    power = (2 * (i // 2)) / np.float32(d_model)
    return position / np.power(base, power)


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# ----------------------Multihead Attention (with scaled dot product)-------------------------
# Utility to create a Dense layer with Glorot initialization (used throughout)
def _get_dense_layer(size, name):
    return tf.keras.layers.Dense(
        size,
        kernel_initializer=tf.keras.initializers.glorot_normal(),
        name=name
    )

# Main Multi-Head Attention class
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # Ensure d_model is divisible by number of heads
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads  # Depth of each attention head

        # Dense layers for projecting inputs to queries (Q), keys (K), and values (V)
        self.wq = _get_dense_layer(d_model, "query")
        self.wk = _get_dense_layer(d_model, "key")
        self.wv = _get_dense_layer(d_model, "value")

        # Final dense layer after concatenating all attention heads
        self.dense = tf.keras.layers.Dense(d_model)

    def call(self, v, k, q, mask, training=False):
        batch_size = tf.shape(q)[0]

        # Step 1: Linear projections (Q, K, V)
        q = self.wq(q, training=training)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k, training=training)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v, training=training)  # (batch_size, seq_len_v, d_model)

        # Step 2: Split into multiple heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Step 3: Scaled Dot-Product Attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        # scaled_attention: (batch_size, num_heads, seq_len_q, depth)

        # Step 4: Combine attention heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # Step 5: Final linear layer
        output = self.dense(concat_attention, training=training)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights  # return both output and attention scores

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension into (num_heads, depth) and
        transposes the result to shape: (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):
        """
        Computes Scaled Dot-Product Attention.

        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth)
          mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k)

        Returns:
          output: attention values
          attention_weights: softmax attention weights
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # Scale by sqrt(depth) to stabilize gradients
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply mask (e.g., to prevent attention to padding or future tokens)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # large negative -> 0 prob after softmax

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth)

        return output, attention_weights


#--------------------Pointwise Feed Forward Network----------------------

class PointwiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        """
        d_model: the dimensionality of the model output (e.g., 512)
        dff: the dimensionality of the inner dense layer (e.g., 2048)
        This is typically much larger than d_model.
        """
        super(PointwiseFeedForwardNetwork, self).__init__()

        # First dense layer expands the dimension to dff and uses ReLU for non-linearity
        self.dense_1 = tf.keras.layers.Dense(dff, activation="relu")

        # Second dense layer projects back to original d_model size
        self.dense_2 = tf.keras.layers.Dense(d_model)

    def call(self, x, training=False):
        """
        Forward pass for feed-forward network
        Applies both dense layers position-wise (same weights applied to each position independently)
        """
        x = self.dense_1(x, training=training)  # (batch_size, seq_len, dff)
        x = self.dense_2(x, training=training)  # (batch_size, seq_len, d_model)
        return x


#----------------------------Encoder Layer-------------------------

# Layer normalization with a small epsilon to avoid divide-by-zero errors
def _get_layer_norm():
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        One encoder layer consists of:
        - Multi-Head Attention (with residual connection + layer norm)
        - Feed Forward Network (with residual connection + layer norm)

        Parameters:
        - d_model: output dimension of model
        - num_heads: number of attention heads
        - dff: dimension of feed-forward network
        - rate: dropout rate
        """
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)  # Self-attention layer
        self.ffn = PointwiseFeedForwardNetwork(d_model, dff)  # Feed-forward network

        self.layernorm1 = _get_layer_norm()  # Norm after attention
        self.layernorm2 = _get_layer_norm()  # Norm after FFN

        self.dropout1 = tf.keras.layers.Dropout(rate)  # Dropout after attention
        self.dropout2 = tf.keras.layers.Dropout(rate)  # Dropout after FFN

    def call(self, x, mask, training=False):
        """
        Forward pass for encoder layer

        x: input tensor of shape (batch_size, input_seq_len, d_model)
        mask: mask for attention (to ignore padding positions)
        """
        # Self-attention block (q = k = v = x)
        attn_output, _ = self.mha(x, x, x, mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        # Residual connection + LayerNorm
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network block
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Residual connection + LayerNorm
        out2 = self.layernorm2(out1 + ffn_output)

        return out2  # (batch_size, input_seq_len, d_model)


#-------------------------Decoder Layer----------------------------

# Layer Normalization helper
def _get_layer_norm():
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        Decoder Layer:
        - 1st Multi-Head Attention: masked self-attention (for target tokens)
        - 2nd Multi-Head Attention: encoder-decoder attention
        - Feed Forward Network
        - Residual connections + Layer Normalization after each block
        - Dropout to prevent overfitting
        """
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)  # Masked self-attention
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # Encoder-decoder attention

        self.ffn = PointwiseFeedForwardNetwork(d_model, dff)  # FFN block

        # Layer normalizations
        self.layernorm1 = _get_layer_norm()
        self.layernorm2 = _get_layer_norm()
        self.layernorm3 = _get_layer_norm()

        # Dropouts after each block
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training=False):
        """
        x: decoder input (batch, target_seq_len, d_model)
        enc_output: encoder output (batch, input_seq_len, d_model)
        look_ahead_mask: prevents seeing future tokens in decoder
        padding_mask: ignores padding tokens in encoder output
        """

        # --- 1. Masked Multi-head Self-Attention ---
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)  # Residual + Norm

        # --- 2. Multi-head Encoder-Decoder Attention ---
        attn2, attn_weights_block2 = self.mha2(
            enc_output,  # value
            enc_output,  # key
            out1,        # query: coming from decoder self-attn
            padding_mask,
            training=training
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # Residual + Norm

        # --- 3. Feed Forward Network ---
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # Residual + Norm

        # Return:
        # - Final output (batch, target_seq_len, d_model)
        # - Attention weights from both MHA blocks (used for visualization or analysis)
        return out3, attn_weights_block1, attn_weights_block2


#----------------------Encoder consisting of multiple EncoderLayer(s)-----------------------

class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,                # Number of EncoderLayers to stack
        d_model,                   # Dimensionality of embeddings
        num_heads,                 # Number of attention heads
        dff,                       # Hidden units in feed-forward network
        input_vocab_size,         # Vocabulary size of input tokens
        maximum_position_encoding,# Max length of input sequences
        rate=0.1                   # Dropout rate
    ):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Word Embedding Layer
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        # Precomputed Positional Encoding (shape: [1, max_pos_enc, d_model])
        self.pos_encoding = positional_encoding(
            maximum_position_encoding,
            self.d_model
        )

        # Stack of EncoderLayers
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, training=False):
        """
        x: (batch_size, input_seq_len)
        mask: Padding mask (optional)
        training: Boolean flag
        """

        seq_len = tf.shape(x)[1]

        # Embed tokens and scale
        x = self.embedding(x)                                     # (batch_size, seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))      # Scale embeddings
        x += self.pos_encoding[:, :seq_len, :]                    # Add positional encoding

        x = self.dropout(x, training=training)

        # Pass through stacked encoder layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask=mask, training=training)

        return x  # Final encoder output (batch_size, seq_len, d_model)


#-----------------------Decoder consisting of multiple DecoderLayer(s)---------------------------

class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,                 # Number of DecoderLayers to stack
        d_model,                    # Dimensionality of embeddings
        num_heads,                  # Number of attention heads
        dff,                        # Hidden units in feed-forward network
        target_vocab_size,         # Vocabulary size of output tokens
        maximum_position_encoding, # Max length of target sequences
        rate=0.1                    # Dropout rate
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)

        self.pos_encoding = positional_encoding(
            maximum_position_encoding,
            d_model
        )

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training=False):
        """
        x: target sequence input (batch_size, target_seq_len)
        enc_output: encoder output (batch_size, input_seq_len, d_model)
        look_ahead_mask: mask to prevent attending to future tokens
        padding_mask: mask to ignore padding in encoder output
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # Token embedding + positional encoding
        x = self.embedding(x)                                     # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))      # Scale embeddings
        x += self.pos_encoding[:, :seq_len, :]                    # Add positional encoding

        x = self.dropout(x, training=training)

        # Pass through stacked decoder layers
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x,
                enc_output,
                look_ahead_mask,
                padding_mask,
                training=training,
            )

            attention_weights[f'decoder_layer{i+1}_block1'] = block1  # Self-attention
            attention_weights[f'decoder_layer{i+1}_block2'] = block2  # Cross-attention

        return x, attention_weights


#---------------------------Transformer-----------------------------

# Metrics for tracking training and evaluation
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        pe_input,
        pe_target,
        rate=0.1
    ):
        super(Transformer, self).__init__()

        # Initialize encoder and decoder
        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            pe_input,
            rate
        )

        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            target_vocab_size,
            pe_target,
            rate
        )

        # Final linear layer that projects decoder outputs to vocab size
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

        # Delay initialization of loss function (optional optimization)
        self._loss_object = None

    def call(self, data, training=False):
        """
        Performs forward pass.
        data: tuple (inp, tar)
        training: training mode flag
        """
        inp, tar = data

        # Create all necessary masks
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self._create_masks(inp, tar)

        # Encoder forward pass
        enc_output = self.encoder(inp,  mask=enc_padding_mask, training=training)

        # Decoder forward pass
        dec_output, attention_weights = self.decoder(
            tar,
            enc_output,
            look_ahead_mask=look_ahead_mask,
            padding_mask=dec_padding_mask,
            training=training
        )

        # Final linear layer to get logits for each vocab token
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

    # Padding mask for encoder and decoder inputs
    def _create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)  # 1 for pad tokens, 0 otherwise
        return seq[:, tf.newaxis, tf.newaxis, :]          # shape: (batch_size, 1, 1, seq_len)

    # Look-ahead mask to prevent decoder from attending to future tokens
    def _create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  # upper triangular matrix
        return mask  # shape: (seq_len, seq_len)

    # Combine padding and look-ahead masks for decoder
    def _create_masks(self, inp, tar):
        enc_padding_mask = self._create_padding_mask(inp)       # for encoder input
        dec_padding_mask = self._create_padding_mask(inp)       # for encoder-decoder attention

        look_ahead_mask = self._create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self._create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    # Training step (used by model.fit() automatically)
    def train_step(self, data):
        inp, tar = data
        tar_inp = tar[:, :-1]     # input to decoder (excluding <eos>)
        tar_real = tar[:, 1:]     # expected output (excluding <sos>)

        with tf.GradientTape() as tape:
            predictions, _ = self((inp, tar_inp), training=True)
            loss = self._loss_function(tar_real, predictions)

        # Compute and apply gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Track loss and accuracy
        train_loss(loss)
        train_accuracy(self._accuracy_function(tar_real, predictions))

        return {
            train_loss.name: train_loss.result(),
            train_accuracy.name: train_accuracy.result(),
        }

    # Custom loss function with padding mask
    def _loss_function(self, real, pred):
        if self._loss_object is None:
            self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction='none'
            )

        # Compute unmasked loss
        loss_ = self._loss_object(real, pred)

        # Apply mask to ignore padding tokens
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)  # average loss

    # Custom accuracy function (ignores padding tokens)
    def _accuracy_function(self, real, pred):
        accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=2), dtype=real.dtype))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    # Required for resetting metrics during each epoch
    @property
    def metrics(self):
        return [train_loss, train_accuracy]

    # Evaluation step
    def test_step(self, data):
        inp, tar = data
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        predictions, _ = self((inp, tar_inp), training=False)
        loss = self._loss_function(tar_real, predictions)

        test_loss(loss)
        test_accuracy(self._accuracy_function(tar_real, predictions))

        return {
            test_loss.name: test_loss.result(),
            test_accuracy.name: test_accuracy.result(),
        }





# Model Architecture Call

transformer = Transformer(
    NUM_LAYERS,
    D_MODEL,
    NUM_HEADS,
    DFF,
    ENCODER_VOCAB_SIZE,
    DECODER_VOCAB_SIZE,
    pe_input=ENCODER_VOCAB_SIZE,
    pe_target=DECODER_VOCAB_SIZE,
)

# Define the directory for saving and loading weights
checkpoint_dir = WEIGHTS_PATH
checkpoint_files = os.listdir(checkpoint_dir)
if len(checkpoint_files) > 0:
    latest_checkpoint = sorted(checkpoint_files, reverse=True)[0]  # Sort in reverse to get the most recent
    # Build the model with dummy inputs of correct shape
    dummy_input = tf.zeros((1, TEXT_LENGTH), dtype=tf.int32)
    dummy_target = tf.zeros((1, SUMMARY_LENGTH), dtype=tf.int32)
    _ = transformer((dummy_input, dummy_target), training=False)  # Run once to build

    transformer.load_weights(os.path.join(checkpoint_dir, latest_checkpoint))
    print(f"Saved weights loaded from {os.path.join(checkpoint_dir, latest_checkpoint)}")
else:
    print("No saved weights found. Starting fresh.")


# Load the pre-trained SentencePiece model
sp = spm.SentencePieceProcessor()
# sp.load(f'{TOKENIZER_DIR}/bengali_spm.model')  # Replace with your SentencePiece model path
sp.load(f'{TOKENIZER_DIR}/bengali_spm.model')

def clean_bengali_text(text):
    """
    Clean Bengali text by removing non-Bengali characters, punctuations, and extra whitespace.
    """
    text = re.sub(r'[^\u0980-\u09FF০-৯\s]', '', text)  # Keep only Bengali characters (Unicode range: U+0980 to U+09FF), digits, and spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces to single space
    return text.strip()  # Strip leading/trailing whitespace

def evaluate(input_document):
    input_ids = sp.encode(input_document, out_type=int)
    input_ids = pad_sequences([input_ids], maxlen=TEXT_LENGTH, padding='post', truncating='post')
    encoder_input = tf.expand_dims(input_ids[0], 0)

    decoder_input = tf.expand_dims([sp.bos_id()], 0)
    output = decoder_input

    for i in range(SUMMARY_LENGTH):
        predictions, attention_weights = transformer((encoder_input, output), training=False)
        predicted_id = tf.argmax(predictions[:, -1:, :], axis=-1, output_type=tf.int32)  # Cast to int32

        if tf.reduce_any(tf.equal(predicted_id, sp.eos_id())):
            break

        output = tf.concat([output, tf.cast(predicted_id, tf.int32)], axis=-1)  # Ensure same type for concatenation

    return tf.squeeze(output[:, 1:], axis=0), attention_weights

def summarize(input_document):
    text = clean_bengali_text(input_document)
    summarized, _ = evaluate(text)
    # summarized, _ = evaluate(input_document=input_document)
    return sp.decode_ids(summarized.numpy().tolist())
