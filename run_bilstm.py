# run_bilstm_attention.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from core.data_processor import DataLoader
from core.model_bilstm import BiLSTMAttentionModel

def plot_confusion(cm, classes, outpath):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    cfg_path = "config.json"
    if not os.path.exists(cfg_path):
        raise FileNotFoundError("config.json not found.")
    configs = json.load(open(cfg_path, 'r'))

    data_cfg = configs.get('data', {})
    model_cfg = configs.get('model', {})
    training_cfg = configs.get('training', {})

    # artifacts folder for bilstm_attention
    save_dir = os.path.join(model_cfg.get('save_dir', 'saved_models'), 'bilstm_attention')
    os.makedirs(save_dir, exist_ok=True)

    data_file = os.path.join('data', data_cfg['filename'])
    seq_len = int(data_cfg.get('sequence_length', 50))
    normalise = bool(data_cfg.get('normalise', True))
    train_frac = float(data_cfg.get('train_test_split', 0.85))

    # DataLoader expected signature: DataLoader(data_path, train_frac, seq_cols=..., label_col=...)
    loader = DataLoader(data_file, train_frac, seq_cols=data_cfg.get('numeric_columns', None), label_col=data_cfg.get('label_col','Phase'))

    vocab_sizes = loader.get_vocab_sizes()
    num_info = loader.get_numeric_info()
    n_numeric = int(num_info.get('n_numeric', 0))
    print("[Run-BiLSTMAttn] Vocab sizes:", vocab_sizes)
    print("[Run-BiLSTMAttn] Numeric features:", num_info.get('numeric_cols', []))

    (X_action_train, X_reg_train, X_self_train, X_num_train), y_train = loader.get_train_data(seq_len=seq_len, normalise=normalise)
    (X_action_test, X_reg_test, X_self_test, X_num_test), y_test = loader.get_test_data(seq_len=seq_len, normalise=normalise)

    print("[Run-BiLSTMAttn] Train windows:", X_action_train.shape, X_num_train.shape, "Test windows:", X_action_test.shape, X_num_test.shape)
    if X_action_train.shape[0] == 0:
        raise RuntimeError("No training windows produced. Reduce sequence_length or check data.")

    timesteps = X_action_train.shape[1]

    # build model
    bilstm_attn = BiLSTMAttentionModel()
    model_cfg['num_classes'] = int(loader.num_classes)
    configs['model'] = model_cfg
    bilstm_attn.build_model(configs, vocab_sizes=vocab_sizes, n_numeric=n_numeric, timesteps=timesteps)

    # class weights
    unique_classes = np.unique(y_train)
    cw = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weight_dict = { int(c): float(w) for c, w in zip(unique_classes, cw) }
    print("[Run-BiLSTMAttn] Class weights:", class_weight_dict)

    X_train_list = [X_action_train, X_reg_train, X_self_train, X_num_train]
    X_test_list = [X_action_test, X_reg_test, X_self_test, X_num_test]

    history = bilstm_attn.train(
        x=X_train_list,
        y=y_train,
        epochs=int(training_cfg.get('epochs', 8)),
        batch_size=int(training_cfg.get('batch_size', 32)),
        save_dir=save_dir,
        validation_split=float(training_cfg.get('validation_split', 0.1)),
        class_weight=class_weight_dict
    )

    # save history
    hist_out = os.path.join(save_dir, "training_history_bilstm_attn.csv")
    pd.DataFrame(history.history).to_csv(hist_out, index=False)
    print("[Run-BiLSTMAttn] Saved training history to:", hist_out)

    # evaluate
    y_pred = bilstm_attn.predict_classes(X_test_list)
    acc = accuracy_score(y_test, y_pred)
    print("\nBiLSTM+Attn Test Accuracy: %.4f" % acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=loader.get_label_mapping()))

    # confusion
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(save_dir, "confusion_matrix_bilstm_attn.png")
    plot_confusion(cm, loader.get_label_mapping(), cm_path)
    print("[Run-BiLSTMAttn] Saved confusion matrix to:", cm_path)

    # save predictions
    out_csv = os.path.join(save_dir, "predictions_vs_true_bilstm_attn.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(out_csv, index=False)
    print("[Run-BiLSTMAttn] Saved predictions CSV to:", out_csv)

    print("[Run-BiLSTMAttn] Completed. Artifacts saved in:", os.path.abspath(save_dir))

if __name__ == '__main__':
    main()
