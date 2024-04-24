import torch
import torch.linalg as LA
from sklearn.utils.extmath import randomized_svd


class FMS:
    """Fast Median Subspace solver

    The optimization objective is the following problem:
        min_{V} ||(I-VV^T) X||_{1,2}  s.t.  V^T V = I
    where:
        V : optimization variable with shape [n_features, n_directions],
            and is constrained to have orthonormal columns
        X : data matrix with shape [n_features, n_samples]
        ||.||_{1,2} : mixed l1/l2 norm for any matrix A is defined by
            ||A||_{1,2} = \\sum_i ||row_i of A||_2

    We solve the problem described above by Fast Median Subspace (FMS) method, 
    which is proposed in the following paper:

    Lerman, G., & Maunu, T. (2018). Fast, robust and non-convex subspace recovery. 
    Information and Inference: A Journal of the IMA, 7(2), 277-336.

    Please refer to the paper for details, and kindly cite the work 
    if you find it is useful.

    Parameters
    ----------
    d : int, required
        The desired dimension of the underlying subspace
    max_iter : int, optional
        The maximum number of iterations
    eps : float, optional
        The safeguard paramter
    tol : float, optional
        The termination tolerance parameter
    no_random : bool, optional
        The parameter controls if randomized SVD is allowed
    spherical: bool, optional
        The parameter controls if normalize the data to unit sphere
    verbose: bool, optional
        The parameter controls if output info during running

    Attributes
    ----------
    V : tensor, shape [n_features, d]
        computed optimization variable in the problem formulation

    Examples
    --------
        See test.py

    Copyright (C) 2022 Tianyu Ding <tianyu.ding0@gmail.com>
    """

    def __init__(
        self,
        max_iter=1000,
        eps=1e-10,
        tol=1e-5,
        no_random=True,
        spherical=False,
        verbose=False,
    ):
        self.max_iter = max_iter
        self.eps = eps
        self.tol = tol
        self.no_random = no_random
        self.spherical = spherical
        self.verbose = verbose

    def run(self, X, d):
        '''
            X: N x D data set with N points dim D
            d: dim of the supspace to find
        '''
        D = X.shape[1]

        if not (0 < d < D):
            raise ValueError("The problem is not well-defined.")

        def loss(V):
            return torch.sum(torch.sqrt(torch.sum(((torch.eye(D)-V @ V.t()) @ X.t()) ** 2, dim=0)))

        if self.spherical:
            # spherize the data
            self.X = X / LA.norm(X, axis=1, keepdims=True)
        else:
            self.X = X

        mn = min(X.shape)

        dist = torch.FloatTensor([1e5])
        iter = 0

        if d > 0.6 * mn or self.no_random is True:
            _, _, Vh = LA.svd(self.X, full_matrices=False)
            Vi = Vh.t()
            Vi = Vi[:, :d]  # D x d
        else:
            Vi, _, _ = randomized_svd(
                self.X.t().numpy(), n_components=d, random_state=0)
            Vi = torch.from_numpy(Vi)

        Vi_prev = Vi
        while dist > self.tol and iter < self.max_iter:

            if self.verbose:
                print('Iter: %3d, L1 loss: %10.3f, dist: %10g'
                      % (iter, loss(Vi_prev).item(), dist.item()))

            # project datapoints onto the orthogonal complement
            C = self.X.t() - Vi @ (Vi.t() @ self.X.t())  # D x N

            scale = LA.norm(C, axis=0, keepdims=True)  # 1 x N

            Y = self.X * torch.min(scale.t()**(-.5), torch.tensor(1/self.eps))

            if d > 0.6 * mn or self.no_random is True:
                _, _, Vh = LA.svd(Y, full_matrices=False)
                Vi = Vh.t()
                Vi = Vi[:, :d]  # D x d
            else:
                Vi, _, _ = randomized_svd(
                    Y.t().numpy(), n_components=d, random_state=0)
                Vi = torch.from_numpy(Vi)

            dist = self.comp_dist(Vi, Vi_prev)

            Vi_prev = Vi
            iter += 1

        return Vi_prev

    def comp_dist(self, S1, S2):
        A = S1.t() @ S2
        U, _, Vh = LA.svd(A)
        Q = U @ Vh
        dist = LA.norm(S2-S1@Q, 'fro') / torch.sqrt(torch.tensor(S1.shape[1]))
        return dist
