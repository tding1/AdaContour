import torch
import torch.linalg as LA
from scipy.linalg import subspace_angles

from PRSGM import *
from FMS import *


def datagen(D, d, N_in, N_out, sigma):
    if N_out > 0:
        outliers = torch.randn(D, N_out)
        outliers /= LA.norm(outliers, axis=0, keepdims=True)

    inliers = torch.cat((torch.randn(d, N_in), torch.zeros(D-d, N_in)), dim=0)
    inliers += sigma * torch.randn(D, N_in)
    inliers /= LA.norm(inliers, axis=0, keepdims=True)

    N = N_in + N_out
    if N_out > 0:
        data = torch.cat((inliers, outliers), dim=1)
    else:
        data = inliers
    data = data[:, torch.randperm(N)]

    return data


def comp_dist(S1, S2):
    A = S1.t() @ S2
    U, _, Vh = LA.svd(A)
    Q = U @ Vh
    dist = LA.norm(S2 - S1@Q, 'fro') / torch.sqrt(torch.tensor(S1.shape[1]))
    return dist


d = 10
D = 30
N_in = 500
N_out = 800
sigma = 0.5

X = datagen(D, d, N_in, N_out, sigma)
gt_U = torch.diag(torch.cat((torch.ones(d), torch.zeros(D-d))))[:, :d]

proj_X = (torch.eye(D) - gt_U @ gt_U.t()) @ X
print('Projection loss: ', torch.norm(proj_X).item())

U_svd, _, _ = LA.svd(X, full_matrices=False)
print('SVD solution to GT: ', comp_dist(gt_U, U_svd[:, :d]).item())

fms = FMS()
U_fms = fms.run(X.t(), d)
print('FMS solution to GT: ', comp_dist(gt_U, U_fms).item())

prsgm = PRSGM()
U_prsgm = prsgm.run(X, d)
print('PRSGM solution to GT: ', comp_dist(gt_U, U_prsgm).item())
