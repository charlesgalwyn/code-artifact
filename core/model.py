# core/model.py
import os
import datetime as dt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Embedding, Concatenate, LSTM, GRU,
                                     Bidirectional, Dense, Dropout, Flatten)
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

class ModelWrapper:
    """
    Keras model wrapper that builds a model accepting:
      - action_seq (batch, timesteps) -> Embedding -> (batch, timesteps, emb_a)
      - reg_seq (batch, timesteps) -> Embedding -> (batch, timesteps, emb_r)
      - self_seq (batch, timesteps) -> Embedding -> (batch, timesteps, emb_s)
      - numeric_seq (batch, timesteps, n_numeric)
    and concatenates along features axis -> recurrent layers -> Dense softmax.
    """

    def __init__(self):
        self.model = None

    def build_model(self, configs, vocab_sizes, n_numeric, timesteps):
        """
        configs: dict from config.json (model-related settings)
        vocab_sizes: dict with keys 'action_vocab','reg_vocab','self_vocab'
        n_numeric: number of numeric features per timestep
        timesteps: number of timesteps (seq_len - 1)
        """
        timer = Timer()
        timer.start()

        model_cfg = configs.get('model', {})
        emb_cfg = model_cfg.get('embeddings', {})  # optional embedding sizes
        recurrent_type = model_cfg.get('rnn_type', 'lstm').lower()  # 'lstm', 'gru', 'bilstm'
        rnn_units = int(model_cfg.get('rnn_units', 128))
        rnn_return_seq = bool(model_cfg.get('rnn_return_seq', False))
        dropout_rate = float(model_cfg.get('dropout', 0.2))
        dense_units = int(model_cfg.get('dense_units', 64))
        optimizer = model_cfg.get('optimizer', 'adam')

        # embedding sizes default heuristics
        def emb_size(vocab):
            # small heuristic: min(50, max(4, vocab//2))
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

        # Embeddings (if vocab ==1 then embedding is trivial but still works)
        action_emb = Embedding(input_dim=max(1, a_vocab), output_dim=a_dim, mask_zero=False, name='action_emb')(action_in)
        reg_emb = Embedding(input_dim=max(1, r_vocab), output_dim=r_dim, mask_zero=False, name='reg_emb')(reg_in)
        self_emb = Embedding(input_dim=max(1, s_vocab), output_dim=s_dim, mask_zero=False, name='self_emb')(self_in)

        # Concatenate embeddings + numeric features
        merged = Concatenate(axis=-1, name='concat_emb_num')([action_emb, reg_emb, self_emb, numeric_in])

        # Recurrent stack
        if recurrent_type == 'gru':
            if model_cfg.get('bidirectional', False):
                rnn = Bidirectional(GRU(rnn_units, return_sequences=rnn_return_seq), name='bidir_gru')(merged)
            else:
                rnn = GRU(rnn_units, return_sequences=rnn_return_seq, name='gru')(merged)
        else:  # default to LSTM family
            if model_cfg.get('bidirectional', False) or recurrent_type == 'bilstm':
                rnn = Bidirectional(LSTM(rnn_units, return_sequences=rnn_return_seq), name='bidir_lstm')(merged)
            else:
                rnn = LSTM(rnn_units, return_sequences=rnn_return_seq, name='lstm')(merged)

        x = rnn
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

        # if rnn returned sequences, flatten/time-distribute final dense; otherwise x is (batch, units)
        if rnn_return_seq:
            # flatten time dimension then dense
            x = tf.keras.layers.Flatten()(x)

        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        num_classes = int(model_cfg.get('num_classes', configs.get('data', {}).get('num_classes', 2)))
        out = Dense(num_classes, activation='softmax', name='out')(x)

        self.model = KModel(inputs=[action_in, reg_in, self_in, numeric_in], outputs=out)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("[Model] Built model with config:", {
            'rnn_type': recurrent_type, 'rnn_units': rnn_units, 'bidirectional': model_cfg.get('bidirectional', False),
            'action_vocab': a_vocab, 'action_dim': a_dim, 'reg_vocab': r_vocab, 'reg_dim': r_dim, 'self_vocab': s_vocab, 'self_dim': s_dim,
            'n_numeric': n_numeric, 'timesteps': timesteps
        })
        self.model.summary()
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir, validation_split=0.1, class_weight=None):
        timer = Timer()
        timer.start()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
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
        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()
        return history

    def predict_classes(self, x):
        probs = self.model.predict(x)
        return probs.argmax(axis=1)

    def predict_proba(self, x):
        return self.model.predict(x)
