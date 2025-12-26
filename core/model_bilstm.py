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
    Attention mechanism for BiLSTM outputs.
    Stores attention weights for interpretability.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_attention_weights = None  # <-- important

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attn_W',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attn_b',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attn_u',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        # inputs: (batch, timesteps, features)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        scores = tf.tensordot(uit, self.u, axes=1)          # (batch, timesteps)
        weights = tf.nn.softmax(scores, axis=1)

        # 🔑 store attention weights for visualization
        self.last_attention_weights = weights

        weighted = inputs * tf.expand_dims(weights, axis=-1)
        context = tf.reduce_sum(weighted, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


class BiLSTMAttentionModel:
    def __init__(self):
        self.model = None
        self.attention_layer = None

    def build_model(self, configs, vocab_sizes, n_numeric, timesteps):
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
        action_emb = Embedding(a_vocab, a_dim, name='action_emb')(action_in)
        reg_emb = Embedding(r_vocab, r_dim, name='reg_emb')(reg_in)
        self_emb = Embedding(s_vocab, s_dim, name='self_emb')(self_in)

        merged = Concatenate(axis=-1)([action_emb, reg_emb, self_emb, numeric_in])

        rnn_out = Bidirectional(
            LSTM(rnn_units, return_sequences=True),
            name='bilstm_seq'
        )(merged)

        # Attention
        self.attention_layer = AttentionLayer(name='attention')
        context = self.attention_layer(rnn_out)

        x = Dense(dense_units, activation='relu')(context)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

        num_classes = int(model_cfg.get(
            'num_classes',
            configs.get('data', {}).get('num_classes', 2)
        ))

        output = Dense(num_classes, activation='softmax', name='out')(x)

        self.model = KModel(
            inputs=[action_in, reg_in, self_in, numeric_in],
            outputs=output
        )

        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model.summary()
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir,
              validation_split=0.1, class_weight=None):

        timer = Timer()
        timer.start()

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f"{dt.datetime.now().strftime('%d%m%Y-%H%M%S')}-bilstm-attn-e{epochs}.h5"
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
        ]

        history = self.model.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weight
        )

        self.model.save(save_path)
        print(f"[BiLSTM+Attn] Model saved: {save_path}")
        timer.stop()
        return history

    def predict_classes(self, x):
        probs = self.model.predict(x)
        return probs.argmax(axis=1)

    def predict_proba(self, x):
        return self.model.predict(x)

    # 🔑 NEW: getter for attention weights
    def get_attention_weights(self):
        if self.attention_layer is None:
            return None
        return self.attention_layer.last_attention_weights
