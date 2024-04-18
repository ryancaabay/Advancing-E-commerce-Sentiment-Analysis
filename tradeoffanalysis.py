import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = {
    "RoBERTa": [0.8601, 0.8597, 0.8620, 0.8597, 0.8547],
    "DistilBERT": [0.8518, 0.8525, 0.8518, 0.8516, 0.8541],
    "DistilRoBERTa": [0.8570, 0.8569, 0.8570, 0.8566, 0.8552]
}
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "CV Score"]

df = pd.DataFrame(data, index=metrics)

fig, ax = plt.subplots()
bar_width = 0.25
index = np.arange(len(df))

bars1 = ax.bar(index, df["RoBERTa"], bar_width, label='RoBERTa')
bars2 = ax.bar(index + bar_width, df["DistilBERT"], bar_width, label='DistilBERT')
bars3 = ax.bar(index + 2 * bar_width, df["DistilRoBERTa"], bar_width, label='DistilRoBERTa')

ax.set_xlabel('Metrics', fontsize=8)
ax.set_ylabel('Scores', fontsize=8)
ax.set_xticks(index + bar_width)
ax.set_xticklabels(df.index, fontsize=8)
ax.tick_params(axis='y', labelsize=8) 
ax.legend(fontsize=8, loc='upper right')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(True, linestyle='--', color='grey', alpha=0.4)

plt.show()