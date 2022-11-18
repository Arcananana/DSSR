import numpy as np
import torch
import utils as util
# for anisotropic kernels
batch_ker = util.random_batch_kernel(
    batch=30000,
    l=21,#l=11 for setting2x2
    sig_min=0.6,
    sig_max=5,
    rate_iso=0,
    scaling=3,
    tensor=False,
    random_disturb=True,
)
'''
#for isotropic kernels
batch_ker = util.random_batch_kernel(
    batch=30000,
    l=21,
    sig_min=0.2,
    sig_max=4,
    rate_iso=1,
    tensor=False,
    random_disturb=False,
)
'''
print("batch kernel shape: {}".format(batch_ker.shape))
b = np.size(batch_ker, 0)
batch_ker = batch_ker.reshape((b, -1))
pca_matrix = util.PCA(batch_ker, k=10).float()
print("PCA matrix shape: {}".format(pca_matrix.shape))
torch.save(pca_matrix, "./pca_matrix.pth")
print("Save PCA matrix at: ./pca_matrix.pth")
