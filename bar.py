import matplotlib.pyplot as plt
import numpy as np
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)


cmap = plt.colormaps['viridis']


fig, ax = plt.subplots(figsize=(2, 6))  
norm = plt.Normalize(vmin=0, vmax=10000)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='vertical')


output_path = 'colorbar_purple_yellow.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

output_path
