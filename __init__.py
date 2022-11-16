import os
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

DATA_DIR = "."

atlas_to_file = {
    "AAL3v1": "AAL3v1_3mm.nii.gz",
    "scheafer400": "scheafer400_HCP_3mm.nii.gz",
    "shen268": "Shen268_HCP_3mm.nii.gz",
    "shen368": "Shen368_HCP_3mm.nii.gz",
}

def show_assignment(atlas_name, cluster_id):
    atlas = nib.load(os.path.join(DATA_DIR, 'atlas', atlas_to_file[atlas_name])).get_fdata().astype('int')
    nonzeros = np.nonzero(atlas)

    xx, yy, zz = np.meshgrid(np.arange(atlas.shape[0]), np.arange(atlas.shape[1]), np.arange(atlas.shape[2]), indexing='ij')

    xx = xx[nonzeros]
    yy = yy[nonzeros]
    zz = zz[nonzeros]
    
    brain_cloud = np.stack([xx, yy, zz], axis=-1)
    brain_cloud[xx < 30, 0] -= 20
    fig = plt.figure(figsize = (24, 4))

    S = np.load(os.path.join(DATA_DIR, "vis_data", atlas_name) + "_50.npy")
    assignment = [S[i-1, cluster_id] for i in atlas[nonzeros]]
    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        if i == 0:
            ax.set_title(f"cluster {cluster_id}", fontsize=20)
        ax.azim = 90 * i + 20
        p = ax.scatter3D(brain_cloud[:, 0], brain_cloud[:, 1], brain_cloud[:, 2], c=assignment, cmap='Reds', edgecolor="black", linewidth=0.2);
        fig.colorbar(p)