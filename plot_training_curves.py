import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_curves(history_csv, model_name, save_dir):
    df = pd.read_csv(history_csv)

    epochs = range(1, len(df) + 1)

    # --- Accuracy plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df['accuracy'], label='Training Accuracy')
    plt.plot(epochs, df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name}: Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_accuracy.png'), dpi=150)
    plt.close()

    # --- Loss plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df['loss'], label='Training Loss')
    plt.plot(epochs, df['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name}: Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss.png'), dpi=150)
    plt.close()

    print(f"[Saved] {model_name} plots")

if __name__ == "__main__":
    models = {
        "LSTM": "saved_models/lstm/training_history.csv",
        "BiLSTM": "saved_models/bilstm/training_history_bilstm.csv",
        "BiLSTM_Attention": "saved_models/bilstm_attention/training_history_bilstm_attn.csv",
        "Transformer": "saved_models/transformer/training_history_transformer.csv",
        "GRU": "saved_models/gru/training_history_gru.csv",
    }

    output_dir = "saved_models/plots"
    os.makedirs(output_dir, exist_ok=True)

    for model_name, path in models.items():
        if os.path.exists(path):
            plot_curves(path, model_name, output_dir)
        else:
            print(f"[Skipped] {model_name} history not found")
