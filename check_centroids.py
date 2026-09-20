"""Count validation images whose root centroid (the center_point prompt) falls on soil.

Run from the folder containing prepared_data/, i.e. the repository root once `sam2_pipeline.py prepare` has been run."""


import numpy as np
from PIL import Image

with open('prepared_data/val_list.txt') as f:
    names = [l.strip() for l in f if l.strip()]

off_root = []
for n in names:
    m = np.array(Image.open(f"prepared_data/masks/{n}/0/00000.png")) > 0
    y, x = np.where(m)
    if len(x) == 0:
        continue
    cx, cy = int(np.mean(x)), int(np.mean(y))
    if not m[cy, cx]:  # arrays are indexed [row, column] = [y, x]
        off_root.append(n)
print(f"{len(off_root)} / {len(names)} centroids fall outside the roots")
