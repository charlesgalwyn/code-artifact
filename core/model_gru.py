# core/model_gru.py
import os
import datetime as dt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, Dropout, GRU, GlobalAveragePooling1D, TimeDistributed
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

class GRUModel:
    def __init__(self):
        self.model = None

    def build_model(self, configs, vocab_sizes, n_numeric, timesteps):
        """
        Build a stacked GRU model that mirrors the other model wrappers.
        - configs: config dict
        - vocab_sizes: dict with 'action_vocab','reg_vocab','self_vocab'
        - n_numeric: number of numeric features per timestep
        - timesteps: int
        """
        timer = Timer()
        timer.start()

        model_cfg = configs.get('model', {})
        embed_dim = int(model_cfg.get('embed_dim', 64))
        gru_units = int(model_cfg.get('gru_units', 128))
        gru_layers = int(model_cfg.get('gru_layers', 2))
        dropout_rate = float(model_cfg.get('dropout', 0.2))
        dense_units = int(model_cfg.get('dense_units', 64))
        optimizer = model_cfg.get('optimizer', 'adam')

        # simple embedding size heuristic
        def emb_size(vocab, default):
            return int(default or max(8, min(50, vocab // 2)))

        a_vocab = int(vocab_sizes.get('action_vocab', 1))
        r_vocab = int(vocab_sizes.get('reg_vocab', 1))
        s_vocab = int(vocab_sizes.get('self_vocab', 1))

        a_dim = int(model_cfg.get('action_dim', emb_size(a_vocab, embed_dim)))
        r_dim = int(model_cfg.get('reg_dim', emb_size(r_vocab, embed_dim//2)))
        s_dim = int(model_cfg.get('self_dim', emb_size(s_vocab, embed_dim//2)))

        # Inputs
        action_in = Input(shape=(timesteps,), dtype='int32', name='action_input')
        reg_in = Input(shape=(timesteps,), dtype='int32', name='reg_input')
        self_in = Input(shape=(timesteps,), dtype='int32', name='self_input')
        numeric_in = Input(shape=(timesteps, n_numeric), dtype='float32', name='numeric_input')

        # Embeddings
        action_emb = Embedding(input_dim=max(1, a_vocab), output_dim=a_dim, mask_zero=False, name='action_emb')(action_in)
        reg_emb = Embedding(input_dim=max(1, r_vocab), output_dim=r_dim, mask_zero=False, name='reg_emb')(reg_in)
        self_emb = Embedding(input_dim=max(1, s_vocab), output_dim=s_dim, mask_zero=False, name='self_emb')(self_in)

        # Concatenate per-timestep vector
        merged = Concatenate(axis=-1, name='concat_emb_num')([action_emb, reg_emb, self_emb, numeric_in])

        x = merged
        # stacked GRU (return_sequences True for intermediate layers)
        for i in range(gru_layers):
            return_sequences = True if i < (gru_layers - 1) else False
            x = GRU(units=gru_units, return_sequences=return_sequences, dropout=dropout_rate, recurrent_dropout=0.0, name=f'gru_{i}')(x)

        # Dense head
        x = Dense(dense_units, activation='relu', name='dense_1')(x)
        x = Dropout(dropout_rate)(x)
        num_classes = int(model_cfg.get('num_classes', configs.get('data', {}).get('num_classes', 2)))
        out = Dense(num_classes, activation='softmax', name='out')(x)

        self.model = KModel(inputs=[action_in, reg_in, self_in, numeric_in], outputs=out)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("[GRU] Built model - gru_units:%s, layers:%s, embed_dim:%s, timesteps:%s" % (gru_units, gru_layers, embed_dim, timesteps))
        self.model.summary()
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir, validation_split=0.1, class_weight=None):
        timer = Timer()
        timer.start()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_fname = os.path.join(save_dir, '%s-gru-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
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
        print('[GRU] Training Completed. Model saved as %s' % save_fname)
        timer.stop()
        return history

    def predict_classes(self, x):
        probs = self.model.predict(x)
        return probs.argmax(axis=1)

    def predict_proba(self, x):
        return self.model.predict(x)
