# plot_model_comparison.py
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Final evaluation metrics (from your experiments)
# ---------------------------------------------------
models = [
    "LSTM",
    "BiLSTM",
    "BiLSTM + Attention",
    "Transformer",
    "GRU"
]

accuracy = [0.83, 0.82, 0.80, 0.82, 0.83]
macro_f1 = [0.78, 0.76, 0.75, 0.76, 0.79]
weighted_f1 = [0.83, 0.81, 0.78, 0.81, 0.83]

# ---------------------------------------------------
# Bar chart: all models with separate bars per metric
# ---------------------------------------------------
x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(x - width, accuracy, width, label='Accuracy')
plt.bar(x, macro_f1, width, label='Macro F1')
plt.bar(x + width, weighted_f1, width, label='Weighted F1')

plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Performance Comparison of Deep Learning Models")
plt.xticks(x, models, rotation=20)
plt.ylim(0.6, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Save figure
os.makedirs("saved_models/comparison_plots", exist_ok=True)
out_path = "saved_models/comparison_plots/model_comparison_bar_chart.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()

print("[Saved] Model comparison bar chart:", out_path)
