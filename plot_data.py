# thanks ai
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Step 1: Gather and parse files
files = [f for f in os.listdir('.') if f.startswith('n_10__k_') and f.endswith('.log')]

data = {}
k_values = set()
p_values = set()

for f in files:
    match = re.search(r'k_(\d+)__p_([0-9.]+)\.log', f)
    if match:
        k = int(match.group(1))
        p = float(match.group(2))
        k_values.add(k)
        p_values.add(p)

        with open(f) as file:
            values = file.read().strip().split(',')
            error = float(values[3])  # select error metric
            data[(k, p)] = error

# Step 2: Sort axes
k_list = sorted(k_values)  # ascending k: 2, 3, 5, 6, 8, 9
p_list = sorted(p_values)  # ascending p: 0.0 ... 0.9

# Step 3: Create heatmap matrix
heatmap = np.full((len(k_list), len(p_list)), np.nan)

for i, k in enumerate(k_list):
    for j, p in enumerate(p_list):
        heatmap[i, j] = data.get((k, p), np.nan)

# Step 4: Plot
plt.figure(figsize=(10, 6))

# Log scale color normalization (handle small errors > 0)
norm = colors.LogNorm(vmin=np.nanmin(heatmap[heatmap > 0]), vmax=np.nanmax(heatmap))

im = plt.imshow(
    heatmap,
    cmap='viridis',
    origin='lower',       # ensure lower k is at bottom
    aspect='auto',
    norm=norm
)

# Step 5: Axis labels
plt.xticks(ticks=np.arange(len(p_list)), labels=[f'{p:.2f}' for p in p_list], rotation=45)
plt.yticks(ticks=np.arange(len(k_list)), labels=k_list)
plt.xlabel('p')
plt.ylabel('k')
plt.title('Log-Scale Heatmap of Error (value[3]) vs k and p')
plt.colorbar(im, label='Avg Error (log scale)')
plt.tight_layout()
plt.show()

