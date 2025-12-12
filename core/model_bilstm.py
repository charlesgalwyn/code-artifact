# core/model_bilstm_attention.py
import os
import datetime as dt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Embedding, Concatenate,
                                     Bidirectional, LSTM, Dense, Dropout, Layer)
from tensorflow.keras.models import Model as KModel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class Timer:
    def __init__(self):
        self._start = None
    def start(self):
        import time
        self._start = time.time()
    def stop(self):
        import time
        if self._start is None:
            return
        print("[Timer] Time taken: %.4f seconds" % (time.time() - self._start))
        self._start = None

class AttentionLayer(Layer):
    """
    Simple attention mechanism for time-distributed RNN outputs.
    Returns context vector (batch_size, units) and attention weights (batch_size, timesteps).
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        self.W = self.add_weight(name='attn_W',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='attn_b',
                                 shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(name='attn_u',
                                 shape=(input_shape[-1],),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs: (batch, timesteps, features)
        # score = u^T tanh(W h_t + b)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)  # (batch, timesteps, features)
        ait = tf.tensordot(uit, self.u, axes=1)  # (batch, timesteps)
        a = tf.nn.softmax(ait, axis=1)  # (batch, timesteps)
        a_expanded = tf.expand_dims(a, axis=-1)  # (batch, timesteps, 1)
        weighted = inputs * a_expanded  # (batch, timesteps, features)
        context = tf.reduce_sum(weighted, axis=1)  # (batch, features)
        return context  # if you need weights, adapt to return tuple

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

class BiLSTMAttentionModel:
    def __init__(self):
        self.model = None

    def build_model(self, configs, vocab_sizes, n_numeric, timesteps):
        """
        configs: config dict
        vocab_sizes: dict('action_vocab','reg_vocab','self_vocab')
        n_numeric: count of numeric features per timestep
        timesteps: sequence length (seq_len)
        """
        timer = Timer()
        timer.start()

        model_cfg = configs.get('model', {})
        emb_cfg = model_cfg.get('embeddings', {})
        rnn_units = int(model_cfg.get('rnn_units', 128))
        dropout_rate = float(model_cfg.get('dropout', 0.2))
        dense_units = int(model_cfg.get('dense_units', 64))
        optimizer = model_cfg.get('optimizer', 'adam')

        def emb_size(vocab):
            return int(emb_cfg.get('default_dim', min(50, max(4, vocab // 2))))

        a_vocab = int(vocab_sizes.get('action_vocab', 1))
        r_vocab = int(vocab_sizes.get('reg_vocab', 1))
        s_vocab = int(vocab_sizes.get('self_vocab', 1))

        a_dim = int(emb_cfg.get('action_dim', emb_size(a_vocab)))
        r_dim = int(emb_cfg.get('reg_dim', emb_size(r_vocab)))
        s_dim = int(emb_cfg.get('self_dim', emb_size(s_vocab)))

        # Inputs
        action_in = Input(shape=(timesteps,), dtype='int32', name='action_input')
        reg_in = Input(shape=(timesteps,), dtype='int32', name='reg_input')
        self_in = Input(shape=(timesteps,), dtype='int32', name='self_input')
        numeric_in = Input(shape=(timesteps, n_numeric), dtype='float32', name='numeric_input')

        # Embeddings
        action_emb = Embedding(input_dim=max(1, a_vocab), output_dim=a_dim, mask_zero=False, name='action_emb')(action_in)
        reg_emb = Embedding(input_dim=max(1, r_vocab), output_dim=r_dim, mask_zero=False, name='reg_emb')(reg_in)
        self_emb = Embedding(input_dim=max(1, s_vocab), output_dim=s_dim, mask_zero=False, name='self_emb')(self_in)

        # Concat features along last axis -> (batch, timesteps, features)
        merged = Concatenate(axis=-1, name='concat_emb_num')([action_emb, reg_emb, self_emb, numeric_in])

        # Bi-directional LSTM returning sequences for attention
        rnn = Bidirectional(LSTM(rnn_units, return_sequences=True), name='bilstm_seq')(merged)

        # Attention
        context = AttentionLayer(name='attention')(rnn)

        # Dense head
        x = Dense(dense_units, activation='relu')(context)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        num_classes = int(model_cfg.get('num_classes', configs.get('data', {}).get('num_classes', 2)))
        out = Dense(num_classes, activation='softmax', name='out')(x)

        self.model = KModel(inputs=[action_in, reg_in, self_in, numeric_in], outputs=out)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("[BiLSTM+Attn] Built model - units:%s, action_vocab:%s, reg_vocab:%s, self_vocab:%s, n_numeric:%s, timesteps:%s" % (
            rnn_units, a_vocab, r_vocab, s_vocab, n_numeric, timesteps
        ))
        self.model.summary()
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir, validation_split=0.1, class_weight=None):
        timer = Timer()
        timer.start()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_fname = os.path.join(save_dir, '%s-bilstm-attn-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]

        history = self.model.fit(
            x, y,
            epochs=int(epochs),
            batch_size=int(batch_size),
            validation_split=float(validation_split),
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weight
        )

        self.model.save(save_fname)
        print('[BiLSTM+Attn] Training Completed. Model saved as %s' % save_fname)
        timer.stop()
        return history

    def predict_classes(self, x):
        probs = self.model.predict(x)
        return probs.argmax(axis=1)

    def predict_proba(self, x):
        return self.model.predict(x)
