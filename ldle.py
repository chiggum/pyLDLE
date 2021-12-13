import pdb

import numpy as np
import copy

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian, minimum_spanning_tree, breadth_first_order
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.linalg import eigh, svd, qr
from scipy.sparse.linalg import eigs, eigsh, svds

from scipy.stats.distributions import chi2
from scipy.linalg import inv, svdvals, orthogonal_procrustes, norm
import custom_procrustes 

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold._locally_linear import null_space

from matplotlib import pyplot as plt

import time

DEBUG = False

# Solves for T, v s.t. T, v = argmin_{R,w)||AR + w - B||_F^2
# Here A and B have same shape n x d, T is d x d and v is 1 x d
def procrustes(A,B,scaling=False):
#     # Use the below snippet of procrustes from custom_procrustes 
#     # which imitates matlab procrustes method
#     d = A.shape[1]
#     a = np.mean(A,0)
#     b = np.mean(B,0)
#     Abar = A-a[np.newaxis,:]
#     Bbar = B-b[np.newaxis,:]
#     T,_ = orthogonal_procrustes(Abar, Bbar)
#     v = b[np.newaxis,:]-np.dot(a[np.newaxis,:],T)
#     err = norm(np.dot(Abar,T) - Bbar)/Abar.shape[0]
#     return T, v, err, 1
    err, _, tform = custom_procrustes.procrustes(B,A,scaling=scaling)
    err = err/A.shape[0]
    return tform['rotation'], tform['translation'], err, tform['scale']

def compute_zeta(d_e_mask, Psi_k_mask):
    if d_e_mask.shape[0]==1:
        return 1
    disc_lip_const = pdist(Psi_k_mask)/squareform(d_e_mask)
    return np.max(disc_lip_const)/np.min(disc_lip_const)

# Computes cost_k, d_k (dest_k)
def cost_of_moving(k, d_e, U, local_param, c, n_C,
                   Utilde, eta_min, eta_max, b=None):
    c_k = c[k]
    # Compute |C_{c_k}|
    n_C_c_k = n_C[c_k]
    
    # Check if |C_{c_k}| < eta_{min}
    # If not then c_k is already
    # an intermediate cluster
    if n_C_c_k >= eta_min:
        return np.inf, -1
    
    # Initializations
    n = n_C.shape[0]    
    cost_x_k_to = np.zeros(n) + np.inf
    
    U_k = U[k,:]==1
    # Compute neighboring clusters c_{U_k} of x_k
    c_U_k = np.unique(c[U_k]).tolist()
    
    # Iterate over all m in c_{U_k}
    for m in c_U_k:
        if m == c_k:
            continue
            
        # Compute |C_{m}|
        n_C_m = n_C[m]
        # Check if |C_{m}| < eta_{max}. If not
        # then mth cluster has reached the max
        # allowed size of the cluster. Move on.
        if n_C_m >= eta_max:
            continue
        
        # Check if |C_{m}| >= |C_{c_k}|. If yes, then
        # mth cluster satisfies all required conditions
        # and is a candidate cluster to x_k in.
        if n_C_m >= n_C_c_k:
            # Compute union of Utilde_m U_k
            Utilde_m = Utilde[m,:]==1
            U_k_U_Utilde_m = U_k | Utilde_m
            # Compute the cost of moving x_k to mth cluster,
            # that is cost_{x_k \rightarrow m}
            cost_x_k_to[m] = compute_zeta(d_e[np.ix_(U_k_U_Utilde_m,U_k_U_Utilde_m)],
                                  local_param.eval_(m, U_k_U_Utilde_m))
        
    
    # find the cluster with minimum cost
    # to move x_k in.
    dest_k = np.argmin(cost_x_k_to)
    cost_k = cost_x_k_to[dest_k]
    if cost_k == np.inf:
        dest_k = -1
        
    return cost_k, dest_k
        
def graph_laplacian(d_e, k_nn, k_tune, gl_type,
                    return_diag=False, use_out_degree=True,
                    tune_type=0):
    if type(k_tune) != list:
        assert k_nn > k_tune, "k_nn must be greater than k_tune."
    assert gl_type in ['normed','unnorm', 'diffusion'],\
            "gl_type should be one of {'normed','unnorm','diffusion'}"
    
    n = d_e.shape[0]
    
    # Find k_nn nearest neighbors
    # If n_neighbors==1 then it nearest neighbor = self
    neigh = NearestNeighbors(n_neighbors=k_nn,
                             metric='precomputed',
                             algorithm='brute')
    neigh.fit(d_e)
    neigh_dist, neigh_ind = neigh.kneighbors()
    
    if type(k_tune) == list:
        epsilon = neigh_dist[:,k_nn-1]
        p, d = k_tune
        autotune = 4*0.5*((epsilon**2)/chi2.ppf(p, df=d))
        autotune = autotune[:,np.newaxis]
    else:
        # Compute tuning values for each pair of neighbors
        sigma = neigh_dist[:,k_tune-1].flatten()
        if tune_type == 0:
            autotune = sigma[neigh_ind]*sigma[:,np.newaxis]
        elif tune_type == 1:
            autotune = sigma[:,np.newaxis]**2
        elif tune_type == 2:
            autotune = np.median(sigma)**2
    
    if tune_type != 3:
        # Compute kernel matrix
        eps = np.finfo(np.float64).eps
        K = np.exp(-neigh_dist**2/(autotune+eps))
    else:
        K = np.ones(neigh_dist.shape)
    
    # Convert to sparse matrices
    source_ind = np.repeat(np.arange(n),k_nn)
    K = coo_matrix((K.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
    autotune = coo_matrix((autotune.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
    K0 = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
    
    
    K = K + K.T
    K0 = K0 + K0.T
    autotune = autotune + autotune.T
    
    K.data /= K0.data
    autotune.data /= K0.data
    
    #pdb.set_trace()
    if gl_type == 'diffusion':
        D = 1/(K.sum(axis=1).reshape((n,1)))
        K = K.multiply(D).multiply(D.transpose())
        gl_type = 'normed'

    # Compute and return graph Laplacian based on gl_type
    if gl_type == 'normed':
        return autotune, neigh_dist, neigh_ind,\
               laplacian(K, normed=True,
                         return_diag=return_diag,
                         use_out_degree=use_out_degree)
    elif gl_type == 'unnorm':
        return autotune, neigh_dist, neigh_ind,\
               laplacian(K, normed=False,
                         return_diag=return_diag,
                         use_out_degree=use_out_degree)
    

def local_views_in_ambient_space(d_e, k, neigh_dist=None):
    if neigh_dist is None:
        neigh = NearestNeighbors(n_neighbors=k,
                                 metric='precomputed',
                                 algorithm='brute')
        neigh.fit(d_e)
        neigh_dist, _ = neigh.kneighbors()
    
    epsilon = neigh_dist[:,[k-1]]
    U = d_e < (epsilon + 1e-12)
    return U, epsilon

def compute_gradient_using_LLR(X, phi, d_e, U, t, d, print_prop = 0.25):
    n,p = X.shape
    print_freq = np.int(n*print_prop)
    N = phi.shape[1]
    grad_phi = np.zeros((n,N,p))
    
    #Uh = d_e < np.sqrt(4*t)
    Uh = d_e < 4*t
    G = np.power(4*t,-d/2)*np.exp(-d_e**2/(4*t))*Uh
    
    for k in range(n):
        if print_freq and np.mod(k,print_freq)==0:
            print('Gradient computation done for %d points...' % k)
            
        U_k = U[k,:]
        n_U_k = np.sum(U_k)
        X_k = X[U_k,:]
        X_k_ = X_k - np.mean(X_k, axis=0)[np.newaxis,:]
        
        Sigma_k = np.dot(X_k_.T, X_k_)/n_U_k
        if p == d:
            _, B_k = eigh(Sigma_k)
        else:
            _, B_k = eigsh(Sigma_k, k=d, which='LM')
        
        Uh_k = Uh[k,:]
        n_Uh_k = np.sum(Uh_k)
        Xh_k = X[Uh_k,:]
        XX_k = np.zeros((n_Uh_k,d+1))
        XX_k[:,0] = 1
        XX_k[:,1:] = np.dot(Xh_k - X[[k],:], B_k)
        
        WW_k = G[k,Uh_k][np.newaxis,:].T
        
        Y = phi[Uh_k,:]
        temp = np.dot(XX_k.T,WW_k*XX_k)
        #print(k)
        #print(XX_k)
        #print(WW_k)
        #print(temp)
        bhat_k = np.dot(np.linalg.inv(temp),np.dot(XX_k.T, WW_k*Y))
        
        grad_phi[k,:,:] = np.dot(B_k,bhat_k[1:,:]).T
    
    return grad_phi

def compute_Atilde(phi, d_e, U, epsilon, p, d, print_prop = 0.25):
    n, N = phi.shape
    print_freq = np.int(n*print_prop)
    
    U = U.copy()
    np.fill_diagonal(U, 0)
    
    # Compute G
    t = 0.5*((epsilon**2)/chi2.ppf(p, df=d))
    #t = 0.5*(np.dot(epsilon.T,epsilon))/chi2.ppf(p, df=d)
    G = np.exp(-d_e**2/(4*t))*U
    G = G/(np.sum(G,1)[:,np.newaxis])

    # Compute Gtilde (Gtilde_k = (1/t_k)[G_{k1},...,G_{kn}])
    Gtilde = G/(2*t) # Helps in correcting issues at the boundary (see page 5  of http://math.gmu.edu/~berry/Publications/VaughnBerryAntil.pdf)
    # Gtilde = G
    
    Atilde=np.zeros((n,N,N))
    
    for k in range(n):
        if print_freq and np.mod(k,print_freq)==0:
            print('A_k, Atilde_k: %d points processed...' % k)
        U_k = U[k,:]==1
        dphi_k = phi[U_k,:]-phi[k,:]
        Atilde[k,:,:] = np.dot(dphi_k.T, dphi_k*(Gtilde[k,U_k][:,np.newaxis]))
    
    print('Atilde_k, Atilde_k: all points processed...')
    return Gtilde, Atilde

def compute_Atilde_LDLE_2(X, L, phi0, phi, lmbda0, lmbda, d_e, U, epsilon, p, d, autotune, print_prop = 0.25):
    n, N = phi.shape
    print_freq = np.int(n*print_prop)
    
    L = L.copy()
    L = L/(autotune.toarray()+1e-12)
    
    lmbda = lmbda.copy()
    lmbda = lmbda.reshape(1,N)
    Atilde=np.zeros((n,N,N))
    
    # For computing derivative at t=0
    for k in range(n):
        if print_freq and np.mod(k,print_freq)==0:
            print('Atilde: : %d points processed...' % k)
        dphi_k = phi-phi[k,:]
        Atilde[k,:,:] = -0.5*np.dot(dphi_k.T, dphi_k*(L[k,:][:,np.newaxis]))
    print('Atilde_k, Atilde_k: all points processed...')
    return None, Atilde

def compute_Atilde_LLR(X, phi, d_e, U, epsilon, p, d, print_prop = 0.25):
    n, N = phi.shape
    print_freq = np.int(n*print_prop)
    
    # t = 0.5*((epsilon**2)/chi2.ppf(p, df=d))
    t = 0.5*((epsilon**2)*chi2.ppf(p, df=d))
    grad_phi = compute_gradient_using_LLR(X, phi, d_e, U, t, d, print_prop = print_prop)
    Atilde=np.zeros((n,N,N))
    
    for k in range(n):
        if print_freq and np.mod(k,print_freq)==0:
            print('A_k, Atilde_k: %d points processed...' % k)
        
        grad_phi_k = grad_phi[k,:,:]
        Atilde[k,:,:] = np.dot(grad_phi_k, grad_phi_k.T)
    
    print('Atilde_k, Atilde_k: all points processed...')
    return grad_phi, Atilde

def compute_Atilde_LDLE_3(X, L, phi0, phi, lmbda0, lmbda, d_e, U, epsilon, p, d, autotune, print_prop = 0.25):
    n, N = phi.shape
    print_freq = np.int(n*print_prop)
    Atilde=np.zeros((n,N,N))
    
    temp1 = np.dot(lmbda*phi, phi.T)
    temp1 = temp1 + np.dot(lmbda0*phi0, phi0.T)
    temp1 = temp1/(autotune.toarray()+1e-12)
    
    for k in range(n):
        if print_freq and np.mod(k,print_freq)==0:
            print('Atilde: : %d eigenvectors processed...' % k)
        
        dphi_k = phi-phi[k,:]
        Atilde[k,:,:] = -0.5*np.dot(dphi_k.T, dphi_k*(temp1[k,:][:,np.newaxis]))
    
    print('Atilde_k, Atilde_k: all points processed...')
    return None, Atilde


def double_manifold(X, ddX):
    d_e = squareform(pdist(X))
    n = X.shape[0]

    dX = ddX==0
    n_dX = np.sum(dX)
    print('No. of points on the boundary =', n_dX)
    d_e_dX1 = d_e[:,dX] # Distance of all the points from boundary
    d_e_dX2 = d_e[np.ix_(~dX,dX)] # Distance of interior points from the boundary
    #d_e_tilde = np.zeros((n,n-n_dX))+np.inf # Distance of doubled interior from the original manifold
    #for k in range(n_dX):
    #    d_e_tilde = np.minimum(d_e_tilde, d_e_dX1[:,k][:,np.newaxis] + d_e_dX2[:,k].T[np.newaxis,:])
    #for i in range(n):
    #    d_e_tilde[i,:] = np.min(d_e_dX1[i,:] + d_e_dX2, axis=1)
    def temp_fn(row):
        return np.min(row[np.newaxis,:] + d_e_dX2, axis=1)

    d_e_tilde = np.apply_along_axis(temp_fn, 1, d_e_dX1)

    d_e_double = np.zeros((2*n-n_dX,2*n-n_dX))
    d_e_double[:n,:n] = d_e
    d_e_double[:n,n:] = d_e_tilde
    d_e_double[n:,:n] = d_e_tilde.T
    d_e_double[n:,n:] = d_e[np.ix_(~dX,~dX)]
    return d_e_double

class Param:
    def __init__(self,
                 algo = 'LDLE',
                 **kwargs):
        self.algo = algo
        self.T = None
        self.v = None
        self.b = None
        
    def eval_(self, k, mask):
        if self.algo == 'LDLE':
            temp = self.Psi_gamma[k,:][np.newaxis,:]*self.phi[np.ix_(mask,self.Psi_i[k,:])]
        elif self.algo == 'LTSA':
#             X_ = self.X[mask,:]
#             temp = np.dot(X_ - np.mean(X_,axis=0)[np.newaxis,:],self.Psi[k,:,:])
            temp = np.dot(self.X[mask,:]-self.mu[k,:][np.newaxis,:],self.Psi[k,:,:])
        if self.b is None:
            return temp
        else:
            temp = self.b[k]*temp
            if self.T is not None and self.v is not None:
                return np.dot(temp, self.T[k,:,:]) + self.v[[k],:]
            else:
                return temp
    
    
class LDLE:
    '''
        X: Input data with examples in rows and features in columns.
        d_e: Dissimilarity matrix.
             Either of X or d_e must be provided.
        k_nn: number of nearest neighbours to consider for graph Laplacian.
        k_tune: self-tuning parameter for graph Laplacian.
        gl_type: The type of graph Laplacian to use {normed, unnorm, random_walk}.
        k: Distance to the kth nearest neighbor is used to
           construct local view in the ambient space.
        p: probability mass to capture.
        d: intrinsic dimension of the manifold.
        tau: percentile for thresholds.
        delta: fraction for thresholds.
        eta_min: minimum number of points in a cluster.
        eta_max: maximum number of points in a cluster.
        to_tear: to tear closed manifolds without boundary or not.
        nu: to account for increased separation.
        max_iter0: Maximum number of iterations to refine global embedding.
        max_iter1: Maximum number of iterations to refine global embedding for a given sequence.
        use_geotorch: Use geotorch to compute final global embedding.
        geotorch_options: options for the geotorch optimizer.
        to_vis_y: whether to visualize global embedding as it is initiated and refined.
    '''
    def __init__(self,
                 X = None,
                 d_e = None,
                 lmbda = None,
                 phi = None,
                 k_nn = 48,
                 k_tune = 6,
                 gl_type = 'unnorm',
                 N = 100,
                 k = 24,
                 no_gamma = False,
                 Atilde_method = 'LDLE_1', 
                 p = 0.99,
                 d = 2,
                 tau = 50,
                 delta = 0.9,
                 to_postprocess = True,
                 eta_min = 5,
                 eta_max = 100,
                 to_tear = True,
                 nu = 3,
                 max_iter0 = 20,
                 max_iter1 = 10,
                 use_geotorch = False,
                 geotorch_options = {'lr': 1e-2, 'n_neg_samples': 5, 'lambda_r': 1e-2},
                 vis = None,
                 vis_y_options = {},
                 local_algo = 'LDLE',
                 global_algo = 'LDLE',
                 ddX = None,
                 exit_at_step1 = False,
                 log_time = False,
                 recompute_d_e = 0,
                 recompute_d_e_t = 1):
        assert X is not None or d_e is not None, "Either X or d_e should be provided."
        self.X = X
        self.d_e = d_e
        self.lmbda = lmbda
        self.phi = phi
        self.k_tune = k_tune
        self.gl_type = gl_type
        self.N = N
        self.k = k
        self.k_nn = max(k_nn,k)
        self.Atilde_method = Atilde_method
        self.no_gamma = no_gamma
        self.p = p
        self.d = d
        self.tau = tau
        self.delta = delta
        self.to_postprocess = to_postprocess
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.to_tear = to_tear
        self.nu = nu
        self.max_iter0 = max_iter0
        self.max_iter1 = max_iter1
        self.use_geotorch = use_geotorch
        self.geotorch_options = geotorch_options
        self.local_algo = local_algo
        self.global_algo = global_algo
        self.ddX = ddX
        self.vis = vis
        self.exit_at_step1 = exit_at_step1
        self.log_time = log_time
        self.recompute_d_e = recompute_d_e
        self.recompute_d_e_t = recompute_d_e_t
        if vis is not None:
            default_vis_y_options = {'cmap0': 'summer',
                                     'cmap1': 'jet',
                                     'labels': np.arange(self.X.shape[0])}
            # Replace default option value with input option value
            for k in vis_y_options:
                default_vis_y_options[k] = vis_y_options[k]
        
            self.vis_y_options = default_vis_y_options
    
    def print_time_log(self,s):
        if self.log_time:
            print(s)
        
    def fit(self):
        old_time0 = time.time()
        old_time = time.time()
        if self.d_e is None:
            if self.ddX is None or self.local_algo == 'LTSA':
                self.d_e = squareform(pdist(self.X))
            else:
                print('Doubling manifold')
                self.d_e = double_manifold(self.X, self.ddX)
                print('Doubled manifold')
        
        self.print_time_log('###############')
        self.print_time_log('Took %0.1f seconds to build distance matrix.' % (time.time()-old_time))
        self.print_time_log('###############')
        old_time = time.time()
        
        if self.local_algo == 'LDLE':
            if self.phi is None:
                # Obtain eigenvalues and eigenvectors of graph Laplacian
                self.lmbda, self.phi = self.eig_graph_laplacian() # also sets self.neigh_ind, self.neigh_dist
            else:
                self.eig_graph_laplacian()
                self.lmbda = np.diagonal(np.dot(self.phi.T, np.dot(self.L,self.phi)))
                print('lmbda:',self.lmbda)

            self.print_time_log('###############')
            self.print_time_log('Took %0.1f seconds to build graph Laplacian and\
                                 compute its eigendecompositon.' % (time.time()-old_time))
            self.print_time_log('###############')
            old_time = time.time()
                
            # Construct local views in the ambient space
            # and obtain radius of each view
            if self.recompute_d_e:
                print('Recomputing d_e using diffusion distance.')
                temp = np.power(self.lmbda[:self.recompute_d_e][np.newaxis,:],self.recompute_d_e_t)
                temp = temp*self.phi[:,:self.recompute_d_e]
                self.d_e = squareform(pdist(temp))
                self.neigh_dist = None
            self.U, epsilon = local_views_in_ambient_space(self.d_e, self.k, self.neigh_dist)
            
            self.print_time_log('###############')
            self.print_time_log('Took %0.1f seconds to construct local views in ambient space.' % (time.time()-old_time))
            self.print_time_log('###############')
            old_time = time.time()

            # Compute Atilde
            if self.Atilde_method == 'LLR':
                print('Using LLR')
                self.grad_phi, self.Atilde = compute_Atilde_LLR(self.X, self.phi, self.d_e, self.U, epsilon, self.p, self.d)
            elif self.Atilde_method == 'LDLE_2':
                print('Using LDLE_2')
                self.grad_phi, self.Atilde = compute_Atilde_LDLE_2(self.X, self.L, self.phi0, self.phi, self.lmbda0, self.lmbda, self.d_e, self.U, epsilon, self.p, self.d, self.autotune)
            elif self.Atilde_method == 'LDLE_3':
                print('Using LDLE_3')
                self.grad_phi, self.Atilde = compute_Atilde_LDLE_3(self.X, self.L, self.phi0, self.phi, self.lmbda0, self.lmbda, self.d_e, self.U, epsilon, self.p, self.d, self.autotune)
            else: #LDLE_1
                print('Using LDLE_1')
                self.Gtilde, self.Atilde = compute_Atilde(self.phi, self.d_e, self.U, epsilon, self.p, self.d)
            
            if self.exit_at_step1:
                return
            
            # Compute gamma
            if self.no_gamma:
                self.gamma = np.ones((self.d_e.shape[0], self.N))
            else:
                self.gamma = np.sqrt(1/(np.dot(self.U,self.phi**2)/np.sum(self.U,1)[:,np.newaxis]))
                
            self.print_time_log('###############')
            self.print_time_log('Took %0.1f seconds to compute Atilde and gamma.' % (time.time()-old_time))
            self.print_time_log('###############')
            old_time = time.time()

            print('\nConstructing low distortion local views using LDLE...')

            # Compute LDLE: Low Distortion Local Eigenmaps
            self.local_param0 = self.compute_LDLE()

            if self.to_postprocess:
                # Postprocess LDLE
                self.local_param = self.postprocess_LDLE()
            else:
                self.local_param = self.local_param0
            
            if self.ddX is not None:
                n = self.X.shape[0]
                self.d_e = self.d_e[:n,:n]
                self.U = self.U[:n,:n]
                self.phi = self.phi[:n,:]
                self.Atilde = self.Atilde[:n,:,:]
                self.gamma = self.gamma[:n,:]
                
                if self.Atilde_method == 'LLR':
                    self.grad_phi = self.grad_phi[:n,:,:]
                
                self.local_param0.Psi_i = self.local_param0.Psi_i[:n,:]
                self.local_param0.Psi_gamma = self.local_param0.Psi_gamma[:n,:]
                self.local_param0.phi = self.local_param0.phi[:n,:]
                self.local_param0.zeta = self.local_param0.zeta[:n]
                
                self.local_param.Psi_i = self.local_param.Psi_i[:n,:]
                self.local_param.Psi_gamma = self.local_param.Psi_gamma[:n,:]
                self.local_param.phi = self.local_param.phi[:n,:]
                self.local_param.zeta = self.local_param.zeta[:n]
                
                for k in range(n):
                    U_k = self.U[k,:]
                    self.local_param0.zeta[k] = compute_zeta(self.d_e[np.ix_(U_k,U_k)], self.local_param0.eval_(k, U_k))
                    self.local_param.zeta[k] = compute_zeta(self.d_e[np.ix_(U_k,U_k)], self.local_param.eval_(k, U_k))
        else:
            # Construct local views in the ambient space
            # and obtain radius of each view
            self.U, epsilon = local_views_in_ambient_space(self.d_e, self.k)

            print('\nConstructing low distortion local views using LTSA...')

            # Compute LDLE: Low Distortion Local Eigenmaps
            self.local_param0 = self.compute_LTSAP()
            
            if self.to_postprocess:
                # Postprocess LDLE
                self.local_param = self.postprocess_LDLE()
            else:
                self.local_param = self.local_param0
        
        self.print_time_log('###############')
        self.print_time_log('Took %0.1f seconds to compute local views in embedding space.' % (time.time()-old_time))
        self.print_time_log('###############')
        old_time = time.time()
        
        print('Max local distortion =', np.max(self.local_param.zeta))
        
        print('###############')
        print('Took %0.1f seconds to perform step 1: construct low distoriton local views' % (time.time()-old_time0))
        print('###############')
        old_time0 = time.time()
        old_time = time.time()
        
        
        # Compute b
        # self.local_param.b = self.compute_b()
            
        print('\nClustering to obtain low distortion intermediate views...')
            
        # Clustering to obtain intermediate views
        self.C, self.c, self.n_C, self.Utilde, self.intermed_param = self.construct_intermediate_views()
        
        print('###############')
        print('Took %0.1f seconds to perform step 2: construct intermediate views.' % (time.time()-old_time0))
        print('###############')
        old_time0 = time.time()
        old_time = time.time()
        
        # Compute final global embedding
        if self.global_algo == 'LDLE':
            self.intermed_param.b = self.compute_b()
            # Compute |Utilde_{mm'}|
            self.n_Utilde_Utilde = np.dot(self.Utilde, self.Utilde.T)
            np.fill_diagonal(self.n_Utilde_Utilde, 0)
            
            self.print_time_log('###############')
            self.print_time_log('Took %0.1f seconds to compute |Utilde_i intersect Utilde_j|.' % (time.time()-old_time))
            self.print_time_log('###############')
            old_time = time.time()
            
            print('\nInitializing parameters and computing initial global embedding...')
        
            # Compute initial global embedding
            self.y_init, self.s_0, self.color_of_pts_on_tear_init = self.compute_initial_global_embedding()
            
            print('###############')
            print('Took %0.1f seconds to compute initial global embedding.' % (time.time()-old_time))
            print('###############')
            old_time = time.time()
            
            print('\nRefining parameters and computing final global embedding...')
            
            print('Using GPA...')
            self.y_final, self.color_of_pts_on_tear_final = self.compute_final_global_embedding(self.max_iter0,
                                                                                                self.max_iter1)
            print('###############')
            print('Took %0.1f seconds to refine global embedding.' % (time.time()-old_time))
            print('###############')
            old_time = time.time()
        elif self.global_algo == 'LTSA':
            print('Using LTSA Global Alignment...')
            self.y_final = self.compute_final_global_embedding_ltsap_based()
            self.color_of_pts_on_tear_final = None
        
        print('###############')
        print('Took %0.1f seconds to perform step 3: compute global embedding.' % (time.time()-old_time0))
        print('###############')
        old_time0 = time.time()
    
    def search_for_tau_and_delta(self, tau_lim=[10,90], ntau=5, delta_lim=[0.1,0.9], ndelta=5):
        if self.d_e is None:
            if self.ddX is None or self.local_algo == 'LTSA':
                self.d_e = squareform(pdist(self.X))
            else:
                print('Doubling manifold')
                self.d_e = double_manifold(self.X, self.ddX)
                print('Doubled manifold')
        # Obtain eigenvalues and eigenvectors of graph Laplacian
        self.lmbda, self.phi = self.eig_graph_laplacian()
        
        # Construct local views in the ambient space
        # and obtain radius of each view
        self.U, epsilon = local_views_in_ambient_space(self.d_e, self.k)
        
        # Compute Atilde
        if self.Atilde_method == 'LLR':
            self.grad_phi, self.Atilde = compute_Atilde_LLR(self.X, self.phi, self.d_e, self.U, epsilon, self.p, self.d)
        elif self.Atilde_method == 'LDLE_2':
            self.grad_phi, self.Atilde = compute_Atilde_formula(self.X, self.L, self.phi, self.lmbda, self.d_e, self.U, epsilon, self.p, self.d)
        else: #LDLE_1
            self.Atilde = compute_Atilde(self.phi, self.d_e, self.U, epsilon, self.p, self.d)
        
        # Compute gamma
        self.gamma = np.sqrt(1/(np.dot(self.U,self.phi**2)/np.sum(self.U,1)[:,np.newaxis]))
        
        print('\nSearching for best tau and delta...')
        
        dtau = (tau_lim[1]-tau_lim[0])/(ntau-1)
        ddelta = (delta_lim[1]-delta_lim[0])/(ndelta-1)
        
        min_max_zeta = np.inf
        tau_star = None
        delta_star = None
        
        self.tau = tau_lim[0] +dtau
        while self.tau <= tau_lim[1]:
            self.delta = delta_lim[0]
            while self.delta <= delta_lim[1]:
                # Compute LDLE: Low Distortion Local Eigenmaps
                self.local_param0 = self.compute_LDLE()

                # Postprocess LDLE
                self.local_param = self.postprocess_LDLE()
                
                
                max_zeta0 = np.max(self.local_param0.zeta)
                max_zeta = np.max(self.local_param.zeta)
                
                print('tau =', self.tau)
                print('delta =', self.delta)
                print('max(zeta0)=', max_zeta0)
                print('max(zeta)=', max_zeta)
                
                if max_zeta < min_max_zeta:
                    tau_star = self.tau
                    delta_star = self.delta
                    min_max_zeta = max_zeta
            
                self.delta += ddelta
            self.tau += dtau
        
        print('Best tau =', tau_star)
        print('Best delta =', delta_star)
        print('Best zeta =', min_max_zeta)
        
    
    def eig_graph_laplacian(self):
        # Eigendecomposition of graph Laplacian
        # Note: Eigenvalues are returned sorted.
        # Following is needed for reproducibility of lmbda and phi
        np.random.seed(42)
        v0 = np.random.uniform(0,1,self.d_e.shape[0])
        if self.gl_type not in ['random_walk', 'diffusion']:
            # Construct graph Laplacian
            self.autotune, self.neigh_dist, self.neigh_ind, L = graph_laplacian(self.d_e, self.k_nn,
                                                                 self.k_tune, self.gl_type)
            lmbda, phi = eigsh(L, k=self.N+1, v0=v0, which='SM')
            L = L.toarray()
        else:
            if self.gl_type == 'random_walk':
                gl_type = 'normed'
            else:
                gl_type = self.gl_type
            # Construct graph Laplacian
            self.autotune, self.neigh_dist, self.neigh_ind, LD = graph_laplacian(self.d_e, self.k_nn,
                                                                    self.k_tune, gl_type,
                                                                    return_diag = True)
            L, D = LD
            lmbda, phi = eigsh(L, k=self.N+1, v0=v0, which='SM')
            L = L.toarray()
            L = (L/D[:,np.newaxis])*D[np.newaxis,:]
            phi = phi/D[:,np.newaxis]
            phi = phi/(np.linalg.norm(phi,axis=0)[np.newaxis,:])
        
        # Ignore the trivial eigenvalue and eigenvector
        self.lmbda0 = lmbda[0]
        self.phi0 = phi[:,0][:,np.newaxis]
        lmbda = lmbda[1:]
        phi = phi[:,1:]
        self.L = L
        self.v0 = v0
        return lmbda, phi
    
    def compute_LDLE(self, print_prop = 0.25):
        # initializations
        n, N = self.phi.shape
        N = self.phi.shape[1]
        d = self.d
        tau = self.tau
        delta = self.delta
        
        print_freq = np.int(n*print_prop)
        
        local_param = Param('LDLE')
        local_param.phi = self.phi
        local_param.Psi_gamma = np.zeros((n,d))
        local_param.Psi_i = np.zeros((n,d),dtype='int')
        local_param.zeta = np.zeros(n)

        # iterate over points in the data
        for k in range(n):
            if print_freq and np.mod(k, print_freq)==0:
                print('local_param: %d points processed...' % k)
            
            # to store i_1, ..., i_d
            i = np.zeros(d, dtype='int')
            
            # Grab the precomputed U_k, Atilde_{kij}, gamma_{ki}
            U_k = self.U[k,:]
            Atilde_k = self.Atilde[k,:,:]
            gamma_k = self.gamma[k,:]
            
            # Compute theta_1
            Atikde_kii = Atilde_k.diagonal()
            theta_1 = np.percentile(Atikde_kii, tau)
            
            # Compute Stilde_k
            Stilde_k = Atikde_kii >= theta_1
            
            # Compute i_1
            r_1 = np.argmax(Stilde_k) # argmax finds first index with max value
            temp = gamma_k * np.abs(Atilde_k[:,r_1])
            alpha_1 = np.max(temp * Stilde_k)
            i[0] = np.argmax((temp >= delta*alpha_1) & (Stilde_k))

            for s in range(1,d):
                i_prev = i[0:s]
                # compute temp variable to help compute Hs_{kij} below
                temp = inv(Atilde_k[np.ix_(i_prev,i_prev)])
                
                # Compute theta_s
                Hs_kii = Atikde_kii - np.sum(Atilde_k[:,i_prev] * np.dot(temp, Atilde_k[i_prev,:]).T, 1)
                temp_ = Hs_kii[Stilde_k]
                theta_s = np.percentile(temp_, tau)
                
                #theta_s=np.max([theta_s,np.min([np.max(temp_),1e-4])])
                
                # Compute i_s
                r_s = np.argmax((Hs_kii>=theta_s) & Stilde_k)
                Hs_kir_s = Atilde_k[:,[r_s]] - np.dot(Atilde_k[:,i_prev], np.dot(temp, Atilde_k[i_prev,r_s][:,np.newaxis]))
                temp = gamma_k * np.abs(Hs_kir_s.flatten())
                alpha_s = np.max(temp * Stilde_k)
                i[s]=np.argmax((temp >= delta*alpha_s) & Stilde_k);
            
            # Compute Psi_k
            local_param.Psi_gamma[k,:] = gamma_k[i]
            local_param.Psi_i[k,:] = i
            
            # Compute zeta_{kk}
            local_param.zeta[k] = compute_zeta(self.d_e[np.ix_(U_k,U_k)], local_param.eval_(k, U_k))
            
        print('local_param: all %d points processed...' % n)
        return local_param
    
    def postprocess_LDLE(self):
        # initializations
        n = self.d_e.shape[0]
        local_param = copy.deepcopy(self.local_param0)
        
        N_replaced = 1
        itr = 1
        # Extra variable to speed up
        param_changed_old = 1
        
        while N_replaced:
            new_param_of = np.arange(n)
            # Extra variable to speed up
            param_changed_new = np.zeros(n)
            # Iterate over all local parameterizations
            for k in range(n):
                U_k = self.U[k,:]
                # To speed up the process, only consider those neighbors
                # for which the parameterization changed in the prev step
                cand_k = np.where(U_k & (param_changed_old==1))[0]
                for kp in cand_k:
                    Psi_kp_on_U_k = local_param.eval_(kp, U_k)
                    zeta_kkp = compute_zeta(self.d_e[np.ix_(U_k,U_k)], Psi_kp_on_U_k)
                    
                    # if zeta_{kk'} < zeta_{kk}
                    if zeta_kkp < local_param.zeta[k]:
                        local_param.zeta[k] = zeta_kkp
                        new_param_of[k] = kp
                        param_changed_new[k] = 1
            
            if self.local_algo == 'LTSA':
                local_param.Psi = local_param.Psi[new_param_of,:,:]
                local_param.mu = local_param.mu[new_param_of,:]
            else:# default to LDLE
                local_param.Psi_i = local_param.Psi_i[new_param_of,:]
                local_param.Psi_gamma = local_param.Psi_gamma[new_param_of,:]
                
            param_changed_old = param_changed_new
            N_replaced = np.sum(param_changed_new)
            
            print("After iter %d, max distortion is %f" % (itr, np.max(local_param.zeta)))
            itr = itr + 1
        
        return local_param
    
    def compute_LTSAP(self, print_prop = 0.25):
        n = self.d_e.shape[0]
        p = self.X.shape[1]
        print_freq = int(print_prop * n)
        # initializations
        local_param = Param('LTSA')
        local_param.X = self.X
        local_param.Psi = np.zeros((n,p,self.d))
        local_param.mu = np.zeros((n,p))
        local_param.zeta = np.zeros(n)

        # iterate over points in the data
        for k in range(n):
            if print_freq and np.mod(k, print_freq)==0:
                print('local_param: %d points processed...' % k)
            
            U_k = self.U[k,:]
            n_U_k = np.sum(U_k)

            # LTSA
            X_k = self.X[U_k,:]
            xbar_k = np.mean(X_k,axis=0)[np.newaxis,:]
            X_k = X_k - xbar_k
            X_k = X_k.T
            if p == self.d:
                Q_k,Sigma_k,_ = svd(X_k)
            else:
                Q_k,Sigma_k,_ = svds(X_k, self.d, which='LM')
                
            local_param.Psi[k,:,:] = Q_k[:,:self.d]
            local_param.mu[k,:] = xbar_k
            
#             C_k = np.dot(X_k.T, X_k)
#             local_param.H.append(eigh(C_k)[1][:, ::-1])
#             local_param.H.append(svd(X_k.T, full_matrices=True)[0])
            
            # Compute zeta_{kk}
            local_param.zeta[k] = compute_zeta(self.d_e[np.ix_(U_k,U_k)], local_param.eval_(k, U_k))
            
        print('local_param: all %d points processed...' % n)
        return local_param
    
#     def compute_b(self):
#         n = self.d_e.shape[0]
#         b = np.zeros(n)
#         for k in range(n):
#             U_k = self.U[k,:]==1
#             d_e_U_k = self.d_e[np.ix_(U_k,U_k)]
#             if d_e_U_k.shape[0]==1:
#                 b[k] = 1
#             else:
#                 Psi_k_on_U_k = self.local_param.eval_(k, U_k)
#                 b[k]=np.median(squareform(d_e_U_k))/np.median(pdist(Psi_k_on_U_k))
#         return b 
    
    def construct_intermediate_views(self):
        # initializations
        n = self.d_e.shape[0]
        c = np.arange(n)
        n_C = np.zeros(n) + 1
        Utilde = np.copy(self.U)
        
        old_time = time.time()
        # Vary eta from 2 to eta_{min}
        for eta in range(2,self.eta_min+1):
            print('# non-empty views with sz < %d = %d' % (eta, np.sum((n_C > 0)*(n_C < eta))))
            print('#nodes in views with sz < %d = %d' % (eta, np.sum(n_C[c]<eta)))
            
            # Compute cost_k and d_k (dest_k) for all k
            cost = np.zeros(n)+np.inf
            dest = np.zeros(n,dtype='int')-1
            for k in range(n):
                cost[k], dest[k] = cost_of_moving(k, self.d_e, self.U, self.local_param,
                                                  c, n_C, Utilde, eta, self.eta_max)
            # Compute point with minimum cost
            # Compute k and cost^* 
            k = np.argmin(cost)
            cost_star = cost[k]
            
            # Loop until minimum cost is inf
            while cost_star < np.inf:
                # Move x_k from cluster s to
                # dest_k and update variables
                s = c[k]
                dest_k = dest[k]
                c[k] = dest_k
                n_C[s] -= 1
                n_C[dest_k] += 1
                Utilde[dest_k,:] = (Utilde[dest_k,:]==1) | (self.U[k,:])
                Utilde[s,:] = np.any(self.U[c==s,:],0)
                
                # Compute the set of points S for which 
                # cost of moving needs to be recomputed
                S = np.where((c==dest_k) | (dest==dest_k) | np.any(self.U[:,c==s],1))[0].tolist()
                
                # Update cost_k and d_k (dest_k) for k in S
                for k in S:
                    cost[k], dest[k] = cost_of_moving(k, self.d_e, self.U, self.local_param,
                                                      c, n_C, Utilde, eta, self.eta_max)
                
                # Recompute point with minimum cost
                # Recompute k and cost^*
                k = np.argmin(cost)
                cost_star = cost[k]
            print('Remaining #nodes in views with sz < %d = %d' % (eta, np.sum(n_C[c]<eta)))
            self.print_time_log('eta = %d, time passed = %0.1f' % (eta, (time.time()-old_time)))
        
        # Prune empty clusters
        non_empty_C = n_C > 0
        M = np.sum(non_empty_C)
        old_to_new_map = np.arange(n)
        old_to_new_map[non_empty_C] = np.arange(M)
        c = old_to_new_map[c]
        n_C = n_C[non_empty_C]
        
        # Construct a boolean array C s.t. C[m,i] = 1 if c_i == m, 0 otherwise
        C = np.zeros((M,n), dtype=bool)
        C[c, np.arange(n)] = True
        
        # Compute intermediate views
        intermed_param = copy.deepcopy(self.local_param)
        if self.local_algo == 'LDLE':
            intermed_param.Psi_i = self.local_param.Psi_i[non_empty_C,:]
            intermed_param.Psi_gamma = self.local_param.Psi_gamma[non_empty_C,:]
        elif self.local_algo == 'LTSA':
            intermed_param.Psi = self.local_param.Psi[non_empty_C,:]
            intermed_param.mu = self.local_param.mu[non_empty_C,:]
            
        # intermed_param.b = self.local_param.b[non_empty_C]
        
        # Compute Utilde_m
        Utilde = np.zeros((M,n),dtype=bool)
        for m in range(M):
            Utilde[m,:] = np.any(self.U[C[m,:],:], 0)
        
        self.Utilde = Utilde
        
        # Compute zetatilde
        intermed_param.zeta = np.zeros(M);
        for m in range(M):
            Utilde_m = Utilde[m,:]
            intermed_param.zeta[m] = compute_zeta(self.d_e[np.ix_(Utilde_m,Utilde_m)],
                                                        intermed_param.eval_(m, Utilde_m))
        
        print("After clustering, max distortion is %f" % (np.max(intermed_param.zeta)))
        return C, c, n_C, Utilde, intermed_param

    def compute_b(self):
        M = self.Utilde.shape[0]
        b = np.zeros(M)
        for m in range(M):
            Utilde_m = self.Utilde[m,:]
            d_e_Utilde_m = self.d_e[np.ix_(Utilde_m,Utilde_m)]
            if d_e_Utilde_m.shape[0]==1:
                b[m] = 1
            else:
                Psitilde_m_on_Utilde_m = self.intermed_param.eval_(m, Utilde_m)
                b[m]=np.median(squareform(d_e_Utilde_m))/np.median(pdist(Psitilde_m_on_Utilde_m))
        return b
    
    def compute_seq_of_intermediate_views(self, print_prop = 0.25):
        M = self.Utilde.shape[0]
        print_freq = int(print_prop * M)
        # First intermediate view in the sequence
        s_1 = np.argmax(self.n_C)

        # Compute |Utilde_{mm'}|
        n_Utilde_Utilde = np.dot(self.Utilde, self.Utilde.T)
        np.fill_diagonal(n_Utilde_Utilde, 0)

        # W_{mm'} = W_{m'm} measures the ambiguity between
        # the pair of the embeddings of the overlap 
        # Utilde_{mm'} in mth and m'th intermediate views
        W = np.zeros((M,M))

        # Iterate over pairs of overlapping intermediate views
        for m in range(M):
            if np.mod(m, print_freq)==0:
                print('Ambiguous overlaps checked for %d intermediate views' % m)
            for mp in np.where(n_Utilde_Utilde[m,:] > 0)[0].tolist():
                if mp > m:
                    # Compute Utilde_{mm'}
                    Utilde_mmp = self.Utilde[m,:]*self.Utilde[mp,:]
                    # Compute V_{mm'}, V_{m'm}, Vbar_{mm'}, Vbar_{m'm}
                    V_mmp = self.intermed_param.eval_(m, Utilde_mmp)
                    V_mpm = self.intermed_param.eval_(mp, Utilde_mmp)
                    Vbar_mmp = V_mmp - np.mean(V_mmp,0)[np.newaxis,:]
                    Vbar_mpm = V_mpm - np.mean(V_mpm,0)[np.newaxis,:]
                    # Compute ambiguity as the minimum singular value of
                    # the d x d matrix Vbar_{mm'}^TVbar_{m'm}
                    W[m,mp] = svdvals(np.dot(Vbar_mmp.T,Vbar_mpm))[-1]
                    W[mp,m] = W[m,mp]

        print('Ambiguous overlaps checked for %d points' % M)
        # Compute maximum spanning tree/forest of W
        T = minimum_spanning_tree(coo_matrix(-W))
        # Detect clusters of manifolds and create
        # a sequence of intermediate views for each of them
        n_visited = 0
        s = []
        rho = []
        # stores cluster number for the intermediate views in a cluster
        Tc = np.zeros(M,dtype=int)
        is_visited = np.zeros(M, dtype=bool)
        c_num = 0
        while n_visited < M:
            # First intermediate view in the sequence
            s_1 = np.argmax(self.n_C * (1-is_visited))
            # Compute breadth first order in T starting from s_1
            s_, rho_ = breadth_first_order(T, s_1, directed=False) #(ignores edge weights)
            s.append(s_)
            rho.append(rho_)
            is_visited[s_] = True
            Tc[s_] = c_num
            n_visited = np.sum(is_visited)
            c_num = c_num + 1
            
        self.Tc = Tc.tolist()
            
            
        print('Seq of intermediate views and their predecessors computed.')
        print('No. of connected components =', len(s))
        if len(s)>1:
            print('Multiple connected components detected')
        return s, rho
    
    def compute_Utildeg(self, y):
        M,n = self.Utilde.shape
        old_time = time.time()
        d_e_ = squareform(pdist(y)) # O(n^2 d)
        self.print_time_log('pdist(y) done. Time taken = %0.1f seconds.' % (time.time()-old_time))
        old_time = time.time()
        Ug, _ = local_views_in_ambient_space(d_e_, np.min([self.k * self.nu, d_e_.shape[0]-1])) # O(n(n+k log(n)))
        self.print_time_log('Ug computed. Time taken = %0.1f seconds.' % (time.time()-old_time))
        old_time = time.time()
        Utildeg = np.zeros((M,n),dtype=bool)
        # O(n^2)
        for m in range(M):
            Utildeg[m,:] = np.any(Ug[self.C[m,:],:], 0)
        
        self.print_time_log('Utildeg computed. Time taken = %0.1f seconds.' % (time.time()-old_time))
        old_time = time.time()
            
        # |Utildeg_{mm'}|
        
        n_Utildeg_Utildeg = np.dot(Utildeg, Utildeg.T) # O(M^2 n) = O(n^3/eta_min^2)
        np.fill_diagonal(n_Utildeg_Utildeg, 0)
        
        self.print_time_log('n_Utildeg_Utildeg computed. Time taken = %0.1f seconds.' % (time.time()-old_time))
            
        return Utildeg, n_Utildeg_Utildeg
       
    # Computes Z_s for the case when to_tear is True.
    # Input Z_s is the Z_s for the case when to_tear is False.
    # Output Z_s is a subset of input Z_s.
    def compute_Z_s_to_tear(self, y, s, Z_s):
        n_Z_s = Z_s.shape[0]
        # C_s_U_C_Z_s = (self.C[s,:]) | np.isin(self.c, Z_s)
        C_s_U_C_Z_s = (self.C[s,:]) | np.any(self.C[Z_s,:], 0)
        c = self.c[C_s_U_C_Z_s]
        n_ = c.shape[0]

        d_e_ = squareform(pdist(y[C_s_U_C_Z_s,:]))
        Ug, _ = local_views_in_ambient_space(d_e_, np.min([self.k,d_e_.shape[0]-1]))
        Utildeg = np.zeros((n_Z_s+1,n_))
        for m in range(n_Z_s):
            Utildeg[m,:] = np.any(Ug[c==Z_s[m],:], 0)

        Utildeg[n_Z_s,:] = np.any(Ug[c==s,:], 0)

        # |Utildeg_{mm'}|
        n_Utildeg_Utildeg = np.dot(Utildeg, Utildeg.T)
        np.fill_diagonal(n_Utildeg_Utildeg, 0)

        return Z_s[n_Utildeg_Utildeg[-1,:-1]>0]
    
    def compute_color_of_pts_on_tear(self, y, n_Utildeg_Utildeg=None):
        M,n = self.Utilde.shape

        # Compute |Utildeg_{mm'}| if not provided
        if n_Utildeg_Utildeg is None:
            _, n_Utildeg_Utildeg = self.compute_Utildeg(y)

        color_of_pts_on_tear = np.zeros(n)+np.nan

        # Compute the tear: a graph between views where ith view
        # is connected to jth view if they are neighbors in the
        # ambient space but not in the embedding space
        tear = (self.n_Utilde_Utilde > 0) & (n_Utildeg_Utildeg == 0)

        # Keep track of visited views across clusters of manifolds
        is_visited = np.zeros(M, dtype=bool)
        n_visited = 0
        while n_visited < M: # boundary of a cluster remain to be colored
            # track the next color to assign
            cur_color = 1

            s0 = np.argmax(is_visited == 0)
            seq, rho = breadth_first_order(self.n_Utilde_Utilde>0, s0, directed=False) #(ignores edge weights)
            is_visited[seq] = True
            n_visited = np.sum(is_visited)

            # Iterate over views
            for m in seq:
                to_tear_mth_view_with = np.where(tear[m,:])[0].tolist()
                if len(to_tear_mth_view_with):
                    # Points in the overlap of mth view and the views
                    # on the opposite side of the tear
                    temp = self.Utilde[m,:][np.newaxis,:] & self.Utilde[to_tear_mth_view_with,:]
                    for i in range(len(to_tear_mth_view_with)):
                        mp = to_tear_mth_view_with[i]
                        # Compute points on the overlap of m and m'th view
                        # which are in mth cluster and in m'th cluster. If
                        # both sets are non-empty then assign them same color.
                        temp_m = temp[i,:] & (self.C[m,:]) & np.isnan(color_of_pts_on_tear)
                        temp_mp = temp[i,:] & (self.C[mp,:])  & np.isnan(color_of_pts_on_tear)
                        if np.any(temp_m) and np.any(temp_mp):
                            color_of_pts_on_tear[temp_m|temp_mp] = cur_color
                            cur_color += 1
                        
        return color_of_pts_on_tear

    def compute_initial_global_embedding(self, print_prop = 0.25):
        old_time = time.time()
        # Intializations
        M,n = self.Utilde.shape
        d = self.d
        print_freq = int(M*print_prop)

        self.intermed_param.T = np.tile(np.eye(d),[M,1,1])
        self.intermed_param.v = np.zeros((M,d))
        err = np.zeros(M)
        y = np.zeros((n,d))

        # Compute the sequence s in which intermediate views
        # are visited. rho[m] = predecessor of mth view.
        all_seq, self.all_rho = self.compute_seq_of_intermediate_views()

        # Boolean array to keep track of already visited views
        is_visited = np.zeros(M, dtype=bool)
        s0_seq = []
        for i in range(len(all_seq)):
            seq = all_seq[i]
            s0_seq.append(seq[0])
            is_visited[seq[0]] = True
            C_s_0 = self.C[seq[0],:]
            y[C_s_0,:] = self.intermed_param.eval_(seq[0], C_s_0)

        self.print_time_log('Naive initialization done. Time passed = %0.1f seconds.' % (time.time()-old_time))

        #pdb.set_trace()

        for i in range(len(all_seq)):
            seq = all_seq[i]
            rho = self.all_rho[i]
            # Traverse views from 2nd view
            for m in range(1,seq.shape[0]):
                if print_freq and np.mod(m, print_freq)==0:
                    print('Initial alignment of %d views completed' % m)
                s = seq[m]
                # pth view is the parent of sth view
                p = rho[s]
                Utilde_s = self.Utilde[s,:]

                # If to tear apart closed manifolds
                if self.to_tear:
                    # Compute T_s and v_s by aligning
                    # the embedding of the overlap Utilde_{sp}
                    # due to sth view with that of the pth view
                    Utilde_s_p = Utilde_s*self.Utilde[p,:]
                    V_s_p = self.intermed_param.eval_(s, Utilde_s_p)
                    V_p_s = self.intermed_param.eval_(p, Utilde_s_p)
                    self.intermed_param.T[s,:,:], self.intermed_param.v[s,:], err[s], _ = procrustes(V_s_p, V_p_s)

                    # Compute temporary global embedding of point in sth cluster
                    C_s = self.C[s,:]
                    y[C_s,:] = self.intermed_param.eval_(s, C_s)

                    # Find more views to align sth view with
                    Z_s = is_visited & (self.n_Utilde_Utilde[s,:]>0)
                    Z_s = np.where(Z_s)[0]
                    Z_s = self.compute_Z_s_to_tear(y, s, Z_s)
                # otherwise
                else:
                    # Align sth view with all the views which have
                    # an overlap with sth view in the ambient space
                    Z_s = is_visited & (self.n_Utilde_Utilde[s,:]>0)
                    Z_s = np.where(Z_s)[0]

                Z_s = Z_s.tolist()
                # If for some reason Z_s is empty
                if len(Z_s)==0:
                    Z_s = [p]

                # Compute centroid mu_s
                # n_Utilde_s_Z_s[k] = #views in Z_s which contain
                # kth point if kth point is in the sth view, else zero
                n_Utilde_s_Z_s = np.zeros(n, dtype=int)
                mu_s = np.zeros((n,d))
                for mp in Z_s:
                    Utilde_s_mp = Utilde_s & self.Utilde[mp,:]
                    n_Utilde_s_Z_s[Utilde_s_mp] += 1
                    mu_s[Utilde_s_mp,:] += self.intermed_param.eval_(mp, Utilde_s_mp)

                # Compute T_s and v_s by aligning the embedding of the overlap
                # between sth view and the views in Z_s, with the centroid mu_s
                temp = n_Utilde_s_Z_s > 0
                mu_s = mu_s[temp,:] / n_Utilde_s_Z_s[temp,np.newaxis]
                V_s_Z_s = self.intermed_param.eval_(s, temp)

                T_s, v_s, err[s], _ = procrustes(V_s_Z_s, mu_s)

                # Update T_s, v_
                self.intermed_param.T[s,:,:] = np.dot(self.intermed_param.T[s,:,:], T_s)
                self.intermed_param.v[s,:] = np.dot(self.intermed_param.v[s,:][np.newaxis,:], T_s) + v_s

                # Mark sth view as visited
                is_visited[s] = True

                # Compute global embedding of point in sth cluster
                C_s = self.C[s,:]
                y[C_s,:] = self.intermed_param.eval_(s, C_s)

        print('error:', np.mean(err))
        
        # refine distance between connected components
        offset = 0
        for i in range(len(all_seq)):
            seq = all_seq[i]
            C_s_ = np.any(self.C[seq,:], axis=0)
            if i > 0:
                offset_ = np.min(y[C_s_,0])
                self.intermed_param.v[seq,0] += offset - offset_
            
            for s in range(seq.shape[0]):
                C_s = self.C[seq[s],:]
                y[C_s,:] = self.intermed_param.eval_(seq[s], C_s)
            
            offset = np.max(y[C_s_,0])
           
        self.print_time_log('Better initialization done. Time passed = %0.1f seconds.' % (time.time()-old_time))

        if self.to_tear:
            color_of_pts_on_tear = self.compute_color_of_pts_on_tear(y)
        else:
            color_of_pts_on_tear = None

        self.print_time_log('Tear computation done. Time passed = %0.1f seconds.' % (time.time()-old_time))

        if self.vis is not None:
            v_opts = self.vis_y_options
            self.vis.global_embedding(y, v_opts['labels'], v_opts['cmap0'],
                                      color_of_pts_on_tear, v_opts['cmap1'],
                                      'Initial')
            plt.show()
            #plt.waitforbuttonpress(1)

        self.print_time_log('Plot done. Time passed = %0.1f seconds.' % (time.time()-old_time))

        return y, s0_seq, color_of_pts_on_tear
    
    def compute_final_global_embedding(self, max_iter0, max_iter1, scaling=False):
        # Intializations
        M,n = self.Utilde.shape
        d = self.d

        np.random.seed(42) # for reproducbility

        y = np.copy(self.y_init)
        
        old_time = time.time()
        # If to tear the closed manifolds
        if self.to_tear:
            # Compute |Utildeg_{mm'}|
            _, n_Utildeg_Utildeg = self.compute_Utildeg(y)
            
        self.print_time_log('n_Utildeg_Utildeg computed. Time passed = %0.1f seconds.' % (time.time()-old_time))

        # Refine global embedding y
        for it0 in range(max_iter0):
            print('Iteration: %d' % it0)

            err = np.zeros(M)
            
            # Traverse over intermediate views in a random order
            seq = np.random.permutation(M)

            # For a given seq, refine the global embedding
            for it1 in range(max_iter1):
                for s in seq.tolist():
                    # Never refine s_0th intermediate view
                    if s in self.s_0:
                        C_s = self.C[s,:]
                        y[C_s,:] = self.intermed_param.eval_(s, C_s)
                        continue

                    Utilde_s = self.Utilde[s,:]

                    # If to tear apart closed manifolds
                    if self.to_tear:
                        # Find more views to align sth view with
                        Z_s = (self.n_Utilde_Utilde[s,:] > 0) & (n_Utildeg_Utildeg[s,:] > 0)
                    # otherwise
                    else:
                        # Align sth view with all the views which have
                        # an overlap with sth view in the ambient space
                        Z_s = self.n_Utilde_Utilde[s,:] > 0

                    Z_s = np.where(Z_s)[0].tolist()
                    
                    if len(Z_s) == 0:
                        Z_s = [self.all_rho[self.Tc[s]][s]]

                    # Compute centroid mu_s
                    # n_Utilde_s_Z_s[k] = #views in Z_s which contain
                    # kth point if kth point is in the sth view, else zero
                    n_Utilde_s_Z_s = np.zeros(n, dtype=int)
                    mu_s = np.zeros((n,d))
                    for mp in Z_s:
                        Utilde_s_mp = Utilde_s & self.Utilde[mp,:]
                        n_Utilde_s_Z_s[Utilde_s_mp] += 1
                        mu_s[Utilde_s_mp,:] += self.intermed_param.eval_(mp, Utilde_s_mp)

                    temp = n_Utilde_s_Z_s > 0
                    mu_s = mu_s[temp,:] / n_Utilde_s_Z_s[temp,np.newaxis]

                    # Compute T_s and v_s by aligning the embedding of the overlap
                    # between sth view and the views in Z_s, with the centroid mu_s
                    V_s_Z_s = self.intermed_param.eval_(s, temp)
                    
                    T_s, v_s, err[s], b_s = procrustes(V_s_Z_s, mu_s, scaling=scaling)

                    # Update T_s, v_s
                    self.intermed_param.T[s,:,:] = np.dot(self.intermed_param.T[s,:,:], T_s)
                    self.intermed_param.v[s,:] = np.dot(self.intermed_param.v[s,:][np.newaxis,:], T_s) + v_s
                    self.intermed_param.b[s] *= b_s

                    # Compute global embedding of points in sth cluster
                    C_s = self.C[s,:]
                    y[C_s,:] = self.intermed_param.eval_(s, C_s)
                    
                self.print_time_log('it0=%d it1=%d completed. Time passed = %0.1f seconds.' % (it0, it1, (time.time()-old_time)))

            # If to tear the closed manifolds
            if self.to_tear:
                # Compute |Utildeg_{mm'}|
                _, n_Utildeg_Utildeg = self.compute_Utildeg(y)
                self.print_time_log('it0=%d, n_Utildeg_Utildeg computed. Time passed = %0.1f seconds.' % (it0, time.time()-old_time))
                color_of_pts_on_tear = self.compute_color_of_pts_on_tear(y, n_Utildeg_Utildeg)
                self.print_time_log('it0=%d, color_of_pts_on_tear computed. Time passed = %0.1f seconds.' % (it0, time.time()-old_time))
            else:
                color_of_pts_on_tear = None
            
            print('error:', np.mean(err))
            # Visualize current embedding
            if self.vis is not None and ((np.mod(it0,5)==0) or (it0==(max_iter0-1))):
                v_opts = self.vis_y_options
                self.vis.global_embedding(y, v_opts['labels'], v_opts['cmap0'],
                                          color_of_pts_on_tear, v_opts['cmap1'],
                                          ('Iter_%d' % it0))
                plt.show()
                #plt.waitforbuttonpress(1)
            
                self.print_time_log('it0=%d, plot done. Time passed = %0.1f seconds.' % (it0, time.time()-old_time))

        return y, color_of_pts_on_tear
    
    def compute_final_global_embedding_ltsap_based(self):
        M,n = self.Utilde.shape
        d = self.d
        
        B = np.zeros((n,n))
        
        for m in range(M):
            Utilde_m = self.Utilde[m,:]
            n_Utilde_m = np.sum(Utilde_m)
            Theta_m = self.intermed_param.eval_(m, Utilde_m)
            Theta_m = Theta_m - np.mean(Theta_m,axis=0)[np.newaxis,:]
            Theta_m = Theta_m.T
            G_mG_mT = 1./n_Utilde_m + np.dot(np.linalg.pinv(Theta_m), Theta_m)
            B[np.ix_(Utilde_m,Utilde_m)] += np.eye(n_Utilde_m) - G_mG_mT
            
#             H_m, _ = qr(Theta_m)
#             H_m = H_m[:,:d][:,::-1]
                  
#             Gm = np.zeros((n_Utilde_m, d + 1))
#             # Gm[:, 1:] = self.intermed_param.H[m][:,:d]
#             Gm[:, 1:] = H_m
#             Gm[:, 0] = 1. / np.sqrt(n_Utilde_m)

#             GmGmT = np.dot(Gm, Gm.T)

#             neighbors = np.where(Utilde_m)[0]
#             nbrs_x, nbrs_y = np.meshgrid(neighbors, neighbors)
#             B[nbrs_x, nbrs_y] -= GmGmT
#             B[neighbors, neighbors] += 1

        y,_ = null_space(B, d, k_skip=1, eigen_solver='auto', random_state=42)
        v_opts = self.vis_y_options
        self.vis.global_embedding(y, v_opts['labels'], v_opts['cmap0'],
                                  None, v_opts['cmap1'],'LTSA')
        plt.waitforbuttonpress(1)
        return y