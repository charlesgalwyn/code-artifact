# core/model_transformer.py
import os
import datetime as dt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Concatenate, Dense, Dropout, Layer, LayerNormalization,
    GlobalAveragePooling1D, MultiHeadAttention, Add
)
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

class PositionalEmbedding(Layer):
    """Learned positional embeddings added to token embeddings."""
    def __init__(self, maxlen, dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.maxlen = int(maxlen)
        self.dim = int(dim)

    def build(self, input_shape):
        # positions 0 .. maxlen-1
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(self.maxlen, self.dim),
            initializer="glorot_uniform",
            trainable=True
        )
        super(PositionalEmbedding, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, dim)
        seq_len = tf.shape(x)[1]
        return x + self.pos_emb[tf.newaxis, :seq_len, :]

    def compute_output_shape(self, input_shape):
        return input_shape

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        attn_out = self.att(inputs, inputs, attention_mask=mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(inputs + attn_out)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.layernorm2(out1 + ffn_out)

class TransformerModel:
    """
    Transformer encoder model wrapper.
    Accepts the same inputs as other models: action, reg, self (ints), numeric (float per timestep).
    """
    def __init__(self):
        self.model = None

    def build_model(self, configs, vocab_sizes, n_numeric, timesteps):
        """
        configs: config dict
        vocab_sizes: dict with 'action_vocab','reg_vocab','self_vocab'
        n_numeric: number of numeric features per timestep
        timesteps: int number of timesteps
        """
        timer = Timer()
        timer.start()

        model_cfg = configs.get('model', {})
        emb_cfg = model_cfg.get('embeddings', {})
        transformer_cfg = model_cfg.get('transformer', {})

        # hyperparams (defaults if not present)
        embed_dim = int(model_cfg.get('embed_dim', 64))
        num_heads = int(transformer_cfg.get('num_heads', 4))
        ff_dim = int(transformer_cfg.get('ff_dim', max(128, embed_dim*2)))
        num_blocks = int(transformer_cfg.get('num_blocks', 2))
        dropout_rate = float(model_cfg.get('dropout', 0.2))
        dense_units = int(model_cfg.get('dense_units', 64))
        optimizer = model_cfg.get('optimizer', 'adam')

        # embedding size heuristics
        def emb_size(vocab, default):
            return int(default or min(50, max(8, vocab//2)))

        a_vocab = int(vocab_sizes.get('action_vocab', 1))
        r_vocab = int(vocab_sizes.get('reg_vocab', 1))
        s_vocab = int(vocab_sizes.get('self_vocab', 1))

        a_dim = int(emb_cfg.get('action_dim', emb_size(a_vocab, embed_dim)))
        r_dim = int(emb_cfg.get('reg_dim', emb_size(r_vocab, embed_dim//2)))
        s_dim = int(emb_cfg.get('self_dim', emb_size(s_vocab, embed_dim//2)))

        # Inputs
        action_in = Input(shape=(timesteps,), dtype='int32', name='action_input')
        reg_in = Input(shape=(timesteps,), dtype='int32', name='reg_input')
        self_in = Input(shape=(timesteps,), dtype='int32', name='self_input')
        numeric_in = Input(shape=(timesteps, n_numeric), dtype='float32', name='numeric_input')

        # Embeddings
        action_emb = Embedding(input_dim=max(1, a_vocab), output_dim=a_dim, mask_zero=False, name='action_emb')(action_in)
        reg_emb = Embedding(input_dim=max(1, r_vocab), output_dim=r_dim, mask_zero=False, name='reg_emb')(reg_in)
        self_emb = Embedding(input_dim=max(1, s_vocab), output_dim=s_dim, mask_zero=False, name='self_emb')(self_in)

        # Concatenate embeddings and numeric to form per-timestep vector
        merged = Concatenate(axis=-1, name='concat_emb_num')([action_emb, reg_emb, self_emb, numeric_in])

        # Project merged to transformer embed_dim if needed
        if merged.shape[-1] != embed_dim:
            proj = Dense(embed_dim, activation='relu', name='proj_to_embed')(merged)
        else:
            proj = merged  # already correct dim

        # Add positional embeddings
        pos = PositionalEmbedding(maxlen=timesteps, dim=embed_dim, name='pos_emb')(proj)

        x = Dropout(dropout_rate)(pos)

        # Transformer encoder blocks
        for i in range(num_blocks):
            x = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate, name=f"transformer_block_{i}")(x)

        # Pooling -> classification head
        x = GlobalAveragePooling1D()(x)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        num_classes = int(model_cfg.get('num_classes', configs.get('data', {}).get('num_classes', 2)))
        out = Dense(num_classes, activation='softmax', name='out')(x)

        self.model = KModel(inputs=[action_in, reg_in, self_in, numeric_in], outputs=out)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("[Transformer] Built model - embed_dim:%s, heads:%s, blocks:%s, ff_dim:%s, timesteps:%s" % (embed_dim, num_heads, num_blocks, ff_dim, timesteps))
        self.model.summary()
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir, validation_split=0.1, class_weight=None):
        timer = Timer()
        timer.start()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_fname = os.path.join(save_dir, '%s-transformer-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
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
        print('[Transformer] Training Completed. Model saved as %s' % save_fname)
        timer.stop()
        return history

    def predict_classes(self, x):
        probs = self.model.predict(x)
        return probs.argmax(axis=1)

    def predict_proba(self, x):
        return self.model.predict(x)
