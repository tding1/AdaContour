import torch
import torch.linalg as LA
from sklearn.utils.extmath import randomized_svd


class PRSGM:
    """Projected Riemannian SubGradient solver

    The optimization objective for RiemannianSubGradient is the following
    least absolute distance problem:
        min_{V} ||(I-VV^T) X||_{1,2}  s.t.  V^T V = I
    where:
        V : optimization variable with shape [n_features, n_directions],
            and is constrained to have orthonormal columns
        X : data matrix with shape [n_features, n_samples]
        ||.||_{1,2} : mixed l1/l2 norm for any matrix A is defined by
            ||A||_{1,2} = \\sum_i ||row_i of A||_2

    We solve the problem described above by Projected Riemannian SubGradient Method
    (PRSGM), which is proposed in the following NeurIPS 2019 paper:

    Zhu, Z., Ding, T., Robinson, D.P., Tsakiris, M.C., & Vidal, R. (2019). 
    A Linearly Convergent Method for Non-Smooth Non-Convex Optimization on the 
    Grassmannian with Applications to Robust Subspace and Dictionary Learning.
    NeurIPS 2019.

    Please refer to the paper for details, and kindly cite the work 
    if you find it is useful.

    Parameters
    ----------
    d : int, required
        The desired dimension of the underlying subspace
    mu_0 : float, optional
        The initial value of step size
    mu_min : float, optional
        The minimum value of step size that is allowed
    max_iter : int, optional
        The maximum number of iterations
    beta : float, optional
        The diminishing factor for step size
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
        mu_0=1e-2,
        mu_min=1e-15,
        max_iter=1000,
        beta=0.8,
        spherical=False,
        verbose=False,
    ):
        self.mu_0 = mu_0
        self.mu_min = mu_min
        self.max_iter = max_iter
        self.beta = beta
        self.spherical = spherical
        self.verbose = verbose

    def run(self, X, d):
        '''
            X: D x N data set with N points dim D
            d: dim of the supspace to find
        '''
        D = X.shape[0]

        if not (0 < d < D):
            raise ValueError("The problem is not well-defined.")

        def loss(V):
            return torch.sum(torch.sqrt(torch.sum(((torch.eye(D)-V @ V.t()) @ X) ** 2, dim=0)))

        if self.spherical:
            # spherize the data
            self.X = X / LA.norm(X, axis=1, keepdims=True)
        else:
            self.X = X

        V, _, _ = LA.svd(self.X, full_matrices=False)
        V = V[:, :d]  # D x d

        mu = self.mu_0
        old_loss = loss(V)
        iter = 0
        while mu > self.mu_min and iter < self.max_iter:

            if self.verbose:
                print('Iter: %3d, L1 loss: %10.3f, mu: %10g'
                      % (iter, old_loss.item(), mu))

            iter += 1

            tmp = torch.sqrt(
                torch.sum(((torch.eye(D)-V @ V.t()) @ self.X) ** 2, dim=0))
            indx = tmp > 0
            grad = -(self.X[:, indx] / tmp[indx]) @ self.X[:, indx].t() @ V
            grad -= V @ (V.t() @ grad)

            # modified line search
            V_next = LA.qr(V - mu * grad)[0]
            while (
                loss(V_next) > old_loss and mu > self.mu_min
            ):
                mu *= self.beta
                V_next = LA.qr(V - mu * grad)[0]
            V = V_next
            old_loss = loss(V)

        return V
