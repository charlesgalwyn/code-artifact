# core/data_processor.py
import os
import numpy as np
import pandas as pd

class DataLoader:
    """
    DataLoader for sequence classification using embeddings for categorical features.
    Features (Option A):
      - ActionShort (embedding)
      - REG_TYPE (embedding)
      - SELF0_PEER1 (embedding / small vocab)
      - Duration -> Duration_log (numeric)
      - LA (numeric)
      - DateCreated -> used to compute time_diff
      - session_pos (numeric)
      - Builds sliding windows and returns inputs as a list:
        [action_seq (N, T), reg_seq (N, T), self_seq (N, T), numeric_seq (N, T, n_numeric)]
      plus y (N,)
    """

    def __init__(self, filename, train_test_split_frac, seq_cols=None, label_col='Phase'):
        """
        filename: path to CSV
        train_test_split_frac: fraction (0..1) of data to use for training (temporal split)
        seq_cols: optional list of numeric columns to include (if None, use defaults below)
        label_col: name of label column
        """
        self.filename = filename
        self.train_test_split_frac = float(train_test_split_frac)
        self.label_col = label_col

        # default numeric columns we will use
        self.numeric_cols = seq_cols or ['Duration', 'LA']

        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Data file not found: {self.filename}")

        self.df = pd.read_csv(self.filename)

        # --- Parse DateCreated robustly ---
        if 'DateCreated' in self.df.columns:
            self.df['DateCreated'] = pd.to_datetime(self.df['DateCreated'], errors='coerce')
            n_before = len(self.df)
            self.df = self.df.dropna(subset=['DateCreated']).reset_index(drop=True)
            n_after = len(self.df)
            if n_after < n_before:
                print(f"[DataLoader] Dropped {n_before - n_after} rows with invalid DateCreated values.")

        # ensure required columns exist (we will be flexible)
        # if numeric cols include 'Duration' we will create Duration_log later
        # ensure label exists
        if self.label_col not in self.df.columns:
            raise KeyError(f"Label column '{self.label_col}' not found in data.")

        # keep only rows that have the label
        self.df = self.df.dropna(subset=[self.label_col]).reset_index(drop=True)

        # --- Duration numeric handling and log transform ---
        if 'Duration' in self.df.columns:
            self.df['Duration'] = pd.to_numeric(self.df['Duration'], errors='coerce')
            n_before = len(self.df)
            self.df = self.df.dropna(subset=['Duration']).reset_index(drop=True)
            n_after = len(self.df)
            if n_after < n_before:
                print(f"[DataLoader] Dropped {n_before - n_after} rows with invalid Duration values.")
            self.df['Duration_log'] = np.log1p(self.df['Duration'])
            # replace numeric col entry 'Duration' with 'Duration_log' in our numeric_cols
            self.numeric_cols = ['Duration_log' if c == 'Duration' else c for c in self.numeric_cols]

        # --- time_diff per session (seconds) ---
        if 'DateCreated' in self.df.columns and 'sessionId' in self.df.columns:
            self.df = self.df.sort_values(['sessionId', 'DateCreated']).reset_index(drop=True)
            self.df['time_diff'] = 0.0
            grouped = self.df.groupby('sessionId')
            for sid, g in grouped:
                if len(g) <= 1:
                    self.df.loc[g.index, 'time_diff'] = 0.0
                    continue
                diffs = g['DateCreated'].diff().dt.total_seconds().fillna(0).values.astype(float)
                self.df.loc[g.index, 'time_diff'] = diffs
            if 'time_diff' not in self.numeric_cols:
                self.numeric_cols.append('time_diff')

        # --- session_pos ---
        if 'sessionId' in self.df.columns:
            self.df['session_pos'] = 0.0
            grouped = self.df.groupby('sessionId')
            for sid, g in grouped:
                n = len(g)
                if n == 1:
                    self.df.loc[g.index, 'session_pos'] = 0.0
                else:
                    positions = np.arange(len(g)) / float(max(1, n - 1))
                    self.df.loc[g.index, 'session_pos'] = positions
            if 'session_pos' not in self.numeric_cols:
                self.numeric_cols.append('session_pos')

        # --- Prepare categorical columns ---
        # ActionShort
        if 'ActionShort' in self.df.columns:
            self.df['ActionShort'] = self.df['ActionShort'].astype(str)
            self.df['ActionShort_code'], self.action_categories = pd.factorize(self.df['ActionShort'])
            self.action_vocab_size = int(self.df['ActionShort_code'].max()) + 1
        else:
            # no ActionShort: create dummy column
            self.df['ActionShort_code'] = 0
            self.action_categories = []
            self.action_vocab_size = 1

        # REG_TYPE
        if 'REG_TYPE' in self.df.columns:
            self.df['REG_TYPE'] = self.df['REG_TYPE'].astype(str)
            self.df['REG_TYPE_code'], self.reg_categories = pd.factorize(self.df['REG_TYPE'])
            self.reg_vocab_size = int(self.df['REG_TYPE_code'].max()) + 1
        else:
            self.df['REG_TYPE_code'] = 0
            self.reg_categories = []
            self.reg_vocab_size = 1

        # SELF0_PEER1 (may be numeric 0/1 or strings)
        if 'SELF0_PEER1' in self.df.columns:
            # coerce to str then factorize to ensure 0/1 mapping even if strings exist
            self.df['SELF0_PEER1'] = self.df['SELF0_PEER1'].astype(str)
            self.df['SELF_code'], self.self_categories = pd.factorize(self.df['SELF0_PEER1'])
            self.self_vocab_size = int(self.df['SELF_code'].max()) + 1
        else:
            # fallback: if there is a column named 'SELF0_PEER1' missing, try 'SELF' or default
            self.df['SELF_code'] = 0
            self.self_categories = []
            self.self_vocab_size = 1

        # --- Label encoding ---
        self.label_values, self.label_classes = pd.factorize(self.df[self.label_col].astype(str))
        self.df['_label_int'] = self.label_values
        self.num_classes = int(len(self.label_classes))

        # --- Convert remaining numeric_cols to numeric (coerce errors and drop rows) ---
        numeric_check_cols = []
        for c in self.numeric_cols:
            if c in self.df.columns:
                numeric_check_cols.append(c)
        # coerce to numeric
        for c in numeric_check_cols:
            self.df[c] = pd.to_numeric(self.df[c], errors='coerce')
        # drop rows with missing numeric values
        n_before = len(self.df)
        if numeric_check_cols:
            self.df = self.df.dropna(subset=numeric_check_cols).reset_index(drop=True)
            n_after = len(self.df)
            if n_after < n_before:
                print(f"[DataLoader] Dropped {n_before - n_after} rows with invalid numeric features.")
        # update numeric cols to those that still exist
        self.numeric_cols = [c for c in self.numeric_cols if c in self.df.columns]

        # --- Train/test temporal split ---
        split_index = int(len(self.df) * float(self.train_test_split_frac))
        if split_index < 1 or split_index >= len(self.df):
            split_index = int(len(self.df) * 0.85)
        self.df_train = self.df.iloc[:split_index].reset_index(drop=True)
        self.df_test = self.df.iloc[split_index:].reset_index(drop=True)

        # Basic info
        print(f"[DataLoader] Train rows: {len(self.df_train)}, Test rows: {len(self.df_test)}")
        print(f"[DataLoader] Action vocab: {self.action_vocab_size}, REG vocab: {self.reg_vocab_size}, SELF vocab: {self.self_vocab_size}")
        print(f"[DataLoader] Numeric cols used: {self.numeric_cols}")

    def _build_seq_inputs(self, df_section, seq_len, normalise=True):
        """
        Build inputs for model:
          action_seq: (N, seq_len-1) int codes
          reg_seq: (N, seq_len-1) int codes
          self_seq: (N, seq_len-1) int codes
          numeric_seq: (N, seq_len-1, n_numeric) floats
          y: (N,)
        """
        data = df_section.reset_index(drop=True)
        n_rows = len(data)
        T = seq_len - 1
        n_numeric = len(self.numeric_cols)

        action_list = []
        reg_list = []
        self_list = []
        numeric_list = []
        y_list = []

        for start in range(0, n_rows - seq_len + 1):
            window = data.loc[start:start + seq_len - 1]  # seq_len rows
            # label is last row's label
            label = int(window.iloc[-1]['_label_int'])
            # integer sequences are the code columns
            action_seq = window['ActionShort_code'].values.astype(int)  # shape (seq_len,)
            reg_seq = window['REG_TYPE_code'].values.astype(int)
            self_seq = window['SELF_code'].values.astype(int)

            # numeric window (seq_len, n_numeric)
            if n_numeric > 0:
                num_window = window[self.numeric_cols].values.astype(float)  # shape (seq_len, n_numeric)
                # per-window normalization across timesteps per column (z-score)
                if normalise:
                    means = num_window.mean(axis=0)
                    stds = num_window.std(axis=0)
                    stds[stds == 0] = 1.0
                    num_window = (num_window - means) / stds
                # take first T timesteps as input
                num_input = num_window[:-1, :]  # (T, n_numeric)
            else:
                # create zeros if no numeric features
                num_input = np.zeros((T, 0), dtype=float)

            # take first T of categorical sequences as input
            action_input = action_seq[:-1].astype(int)
            reg_input = reg_seq[:-1].astype(int)
            self_input = self_seq[:-1].astype(int)

            action_list.append(action_input)
            reg_list.append(reg_input)
            self_list.append(self_input)
            numeric_list.append(num_input)
            y_list.append(label)

        if len(action_list) == 0:
            # return empty arrays shaped correctly
            return [np.empty((0, T), dtype=int),
                    np.empty((0, T), dtype=int),
                    np.empty((0, T), dtype=int),
                    np.empty((0, T, n_numeric), dtype=float)], np.empty((0,), dtype=int)

        X_action = np.vstack([a.reshape(1, -1) for a in action_list])
        X_reg = np.vstack([a.reshape(1, -1) for a in reg_list])
        X_self = np.vstack([a.reshape(1, -1) for a in self_list])
        X_numeric = np.stack(numeric_list, axis=0)  # (N, T, n_numeric)
        y = np.array(y_list, dtype=int)

        return [X_action, X_reg, X_self, X_numeric], y

    def get_train_data(self, seq_len=50, normalise=True):
        return self._build_seq_inputs(self.df_train, seq_len, normalise=normalise)

    def get_test_data(self, seq_len=50, normalise=True):
        return self._build_seq_inputs(self.df_test, seq_len, normalise=normalise)

    def get_label_mapping(self):
        return list(self.label_classes)

    def get_vocab_sizes(self):
        return {
            'action_vocab': int(self.action_vocab_size),
            'reg_vocab': int(self.reg_vocab_size),
            'self_vocab': int(self.self_vocab_size)
        }

    def get_numeric_info(self):
        return {
            'numeric_cols': self.numeric_cols,
            'n_numeric': len(self.numeric_cols)
        }
