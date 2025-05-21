```python
import matplotlib.pyplot as plt
import os

output_dir = '/content/drive/MyDrive/results/'
os.makedirs(output_dir, exist_ok=True)


models = ['Baseline (12 leads)', 'SMFB-Net (9 leads)', 'SMFB-Net (9.8 leads)', 'SMFB-Net (9.8 leads, LSM)']
flops = [2.566, 1.925, 1.968, 1.771]  # Gigaflops
params = [680.457, 510.345, 521.686, 521.686]  # Kilo parameters
reductions_flops = [0, 24.98, 23.31, 30.98]  # % reduction in FLOPs
reductions_params = [0, (1 - 510.345 / 680.457) * 100, (1 - 521.686 / 680.457) * 100, (1 - 521.686 / 680.457) * 100]  # % reduction in params

plt.figure(figsize=(10, 6))
bars = plt.bar(models, flops, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylabel('FLOPs (Gigaflops)')
plt.title('Computational Efficiency: SMFB-Net vs. Baseline (FLOPs)')
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, bar in enumerate(bars):
    height = bar.get_height()
    if i > 0:  # Skip baseline
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.05, f'{reductions_flops[i]:.2f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'flops_comparison.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved FLOPs plot to {os.path.join(output_dir, 'flops_comparison.pdf')}")

plt.figure(figsize=(10, 6))
bars = plt.bar(models, params, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylabel('Parameters (Kilo)')
plt.title('Model Complexity: SMFB-Net vs. Baseline (Parameters)')
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, bar in enumerate(bars):
    height = bar.get_height()
    if i > 0:  # Skip baseline
        plt.text(bar.get_x() + bar.get_width() / 2, height + 20, f'{reductions_params[i]:.2f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'params_comparison.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved Parameters plot to {os.path.join(output_dir, 'params_comparison.pdf')}")

print(f"Listing files in {output_dir}:")
print(os.listdir(output_dir))
```