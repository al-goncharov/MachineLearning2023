import numpy as np
import scipy
from scipy.special import expit
from scipy.sparse import csr_matrix, diags


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        
    def func(self, x):
        Ax = self.matvec_Ax(x)
        m = Ax.shape[0]
        
        return np.mean(np.logaddexp(np.zeros(m), -self.b * Ax)) + 0.5 * self.regcoef * x.dot(x)
    

    def grad(self, x):
        Ax = self.matvec_Ax(x)
        m = Ax.shape[0]
        sigm = expit(-self.b * Ax)
        ATx = self.matvec_ATx(self.b * sigm)
        
        return -1/m * ATx + self.regcoef * x
    
    def hess(self, x):
        Ax = self.matvec_Ax(x)
        m = Ax.shape[0]
        n = x.shape[0]
        sigm = expit(self.b * Ax)
        ATsA = self.matmat_ATsA(sigm * (1 - sigm))
        
        return 1/m * ATsA + self.regcoef * np.eye(n)
    

class LogRegL2OptimizedOracle(LogRegL2Oracle):

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    if scipy.sparse.issparse(A):
        def matvec_Ax(x):
            return A @ x

        def matvec_ATx(x):
            return A.T @ x

        def matmat_ATsA(s):
            As = A.multiply(s.reshape(-1, 1))
            return A.T @ As
    else:
        def matvec_Ax(x):
            return np.dot(A, x)

        def matvec_ATx(x):
            return np.dot(A.T, x)

        def matmat_ATsA(s):
            return A.T.dot(A * s.reshape(-1, 1))

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-5):

    n = x.shape[0]
    grad_finite = np.zeros(n)

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        grad_finite[i] = (func(x + eps * e_i) - func(x)) / eps
    
    return grad_finite


def hess_finite_diff(func, x, eps=1e-5):

    n = x.shape[0]
    hess_finite = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            e_i = np.zeros(n)
            e_j = np.zeros(n)
            e_i[i] = 1
            e_j[j] = 1
            hess_finite[i][j] = (func(x + eps * e_i + eps * e_j)
                                - func(x + eps * e_i)
                                - func(x + eps * e_j)
                                + func(x)) / (eps ** 2)
                
    return hess_finite
