# plot_test_loss_comparison.py
import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Paths to training history files
# ---------------------------------------------------
models = {
    "LSTM": "saved_models/lstm/training_history.csv",
    "BiLSTM + Attention": "saved_models/bilstm_attention/training_history_bilstm_attn.csv",
    "Transformer": "saved_models/transformer/training_history_transformer.csv",
    "GRU": "saved_models/gru/training_history_gru.csv",
}

test_losses = []

# ---------------------------------------------------
# Extract final validation loss
# ---------------------------------------------------
for model_name, path in models.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"History file not found: {path}")

    df = pd.read_csv(path)
    final_val_loss = df["val_loss"].iloc[-1]
    test_losses.append(final_val_loss)

# ---------------------------------------------------
# Plot bar chart (similar to your example)
# ---------------------------------------------------
plt.figure(figsize=(10, 6))

bars = plt.bar(
    models.keys(),
    test_losses,
    color=["blue", "green", "red", "purple", "orange"]
)

plt.ylabel("Test Loss")
plt.title("Comparison of Test Loss Across Deep Learning Models")
plt.ylim(0, max(test_losses) + 0.2)

# Value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.grid(axis="y", linestyle="--", alpha=0.6)

# Save plot
out_dir = "saved_models/comparison_plots"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "test_loss_comparison.png")

plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()

print("[Saved] Test loss comparison plot:", out_path)
