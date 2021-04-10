import pdb

import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian, minimum_spanning_tree, breadth_first_order
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigs, eigsh
from scipy.stats.distributions import chi2
from scipy.linalg import inv, svdvals, orthogonal_procrustes, norm

from sklearn.neighbors import NearestNeighbors

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import geotorch

# Solves for T, v s.t. T, v = argmin_{R,w)||AR + w - B||_F^2
# Here A and B have same shape n x d, T is d x d and v is 1 x d
def procrustes(A,B):
    d = A.shape[1]
    a = np.mean(A,0)
    b = np.mean(B,0)
    Abar = A-a[np.newaxis,:]
    Bbar = B-b[np.newaxis,:]
    T,_ = orthogonal_procrustes(Abar, Bbar)
    v = b[np.newaxis,:]-np.dot(a[np.newaxis,:],T)
    err = norm(np.dot(Abar,T) - Bbar)/Abar.shape[0]
    return T, v, err

def eval_param(phi, Psi_gamma, Psi_i, k, mask, beta=None, T=None, v=None):
    if beta is None:
        return Psi_gamma[k,:][np.newaxis,:] * phi[np.ix_(mask,Psi_i[k,:])]
    else:
        if T is not None and v is not None:
            return np.dot(beta[k]*Psi_gamma[k,:][np.newaxis,:] * phi[np.ix_(mask,Psi_i[k,:])], T[k,:,:]) + v[[k],:]
        else:
            return beta[k]*Psi_gamma[k,:][np.newaxis,:] * phi[np.ix_(mask,Psi_i[k,:])]

def compute_zeta(d_e_mask, Psi_k_mask):
    if d_e_mask.shape[0]==1:
        return 1
    disc_lip_const = pdist(Psi_k_mask)/squareform(d_e_mask)
    return np.max(disc_lip_const)/np.min(disc_lip_const)

# Computes cost_k, d_k (dest_k)
def cost_of_moving(k, d_e, U, phi, Psi_gamma, Psi_i, c,
                    n_C, Utilde, eta_min, eta_max, beta=None):
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
                                  eval_param(phi, Psi_gamma, Psi_i, m, U_k_U_Utilde_m, beta))
        
    
    # find the cluster with minimum cost
    # to move x_k in.
    dest_k = np.argmin(cost_x_k_to)
    cost_k = cost_x_k_to[dest_k]
    if cost_k == np.inf:
        dest_k = -1
        
    return cost_k, dest_k
        
def graph_laplacian(d_e, k_nn, k_tune, gl_type,
                    return_diag=False, use_out_degree=False):
    assert k_nn > k_tune, "k_nn must be greater than k_tune."
    assert gl_type in ['normed','unnorm','random_walk'],\
            "gl_type should be one of {'normed','unnorm','random_walk'}"
    
    n = d_e.shape[0]
    # Find k_nn nearest neighbors excluding self
    neigh = NearestNeighbors(n_neighbors=k_nn,
                             metric='precomputed',
                             algorithm='brute')
    neigh.fit(d_e)
    neigh_dist, neigh_ind = neigh.kneighbors()
    
    # Compute tuning values for each pair of neighbors
    sigma = neigh_dist[:,k_tune-1].flatten()
    autotune = sigma[neigh_ind]*sigma[:,np.newaxis]
    
    # Compute kernel matrix
    eps = np.finfo(np.float64).eps
    K = np.exp(-neigh_dist**2/(autotune+eps))
    
    # Convert to sparse matrices
    neigh_ind = neigh_ind.flatten()
    source_ind = np.repeat(np.arange(n),k_nn)
    K = coo_matrix((K.flatten(),(source_ind,neigh_ind)),shape=(n,n))

    # Compute and return graph Laplacian based on gl_type
    if gl_type == 'normed':
        return laplacian(K, normed=True,
                         return_diag=return_diag,
                         use_out_degree=use_out_degree)
    elif gl_type == 'unnorm':
        return laplacian(K, normed=False,
                         return_diag=return_diag,
                         use_out_degree=use_out_degree)
        return L, D
    

def local_views_in_ambient_space(d_e, k):
    neigh = NearestNeighbors(n_neighbors=k,
                             metric='precomputed',
                             algorithm='brute')
    neigh.fit(d_e)
    neigh_dist, neigh_ind = neigh.kneighbors()
    epsilon = neigh_dist[:,[k-1]]
    U = d_e < (epsilon + 1e-12)
    return U, epsilon

def compute_Atilde(phi, d_e, U, epsilon, p, d, print_prop = 0.25):
    n, N = phi.shape
    print_freq = np.int(n*print_prop)
    
    # Compute G
    t = 0.5*((epsilon**2)/chi2.ppf(p, df=d))
    G = np.exp(-d_e**2/(4*t))*U
    G = G/(np.sum(G,1)[:,np.newaxis])

    # Compute Gtilde (Gtilde_k = (1/t_k)[G_{k1},...,G_{kn}])
    Gtilde = G/(2*t)
    
    Atilde=np.zeros((n,N,N))
    
    for k in range(n):
        if print_freq and np.mod(k,print_freq)==0:
            print('A_k, Atilde_k: %d points processed...' % k)
        U_k = U[k,:]==1
        dphi_k = phi[U_k,:]-phi[k,:]
        Atilde[k,:,:] = np.dot(dphi_k.T, dphi_k*(Gtilde[k,U_k][:,np.newaxis]))
    
    print('Atilde_k, Atilde_k: all points processed...')
    return Atilde
    
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
                 k_nn = 48,
                 k_tune = 6,
                 gl_type = 'unnorm',
                 N = 100,
                 k = 24,
                 p = 0.99,
                 d = 2,
                 tau = 50,
                 delta = 0.9,
                 eta_min = 5,
                 eta_max = 100,
                 to_tear = True,
                 nu = 3,
                 max_iter0 = 20,
                 max_iter1 = 10,
                 use_geotorch = False,
                 geotorch_options = {'lr': 1e-2},
                 vis = None,
                 vis_y_options = {}):
        assert X is not None or d_e is not None, "Either X or d_e should be provided."
        self.X = X
        self.d_e = d_e
        if d_e is None:
            self.d_e = squareform(pdist(X))
        self.k_nn = k_nn
        self.k_tune = k_tune
        self.gl_type = gl_type
        self.N = N
        self.k = k
        self.p = p
        self.d = d
        self.tau = tau
        self.delta = delta
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.to_tear = to_tear
        self.nu = nu
        self.max_iter0 = max_iter0
        self.max_iter1 = max_iter1
        self.use_geotorch = use_geotorch
        self.geotorch_options = geotorch_options
        self.vis = vis
        if vis is not None:
            default_vis_y_options = {'cmap0': 'summer',
                                     'cmap1': 'jet',
                                     'labels': np.arange(self.d_e.shape[0])}
            # Replace default option value with input option value
            for k in vis_y_options:
                default_vis_y_options[k] = vis_y_options[k]
        
            self.vis_y_options = default_vis_y_options
        
        # Eigendecomposition of graph Laplacian
        # Note: Eigenvalues are returned sorted.
        # Following is needed for reproducibility of lmbda and phi
        np.random.seed(42)
        v0 = np.random.uniform(0,1,self.d_e.shape[0])
        if self.gl_type != 'random_walk':
            # Construct graph Laplacian
            self.L = graph_laplacian(self.d_e, self.k_nn,
                                     self.k_tune, self.gl_type)
            self.lmbda, self.phi = eigsh(self.L, k=self.N+1, v0=v0, which='SM')
        else:
            # Construct graph Laplacian
            self.L, self.D = graph_laplacian(self.d_e, self.k_nn,
                                             self.k_tune, 'unnorm', return_diag = True)
            self.lmbda, self.phi = eigs(self.L, k=self.N+1, M=diags(self.D), v0=v0, which='SM')
        
        # Ignore the trivial eigenvalue and eigenvector
        self.lmbda = self.lmbda[1:]
        self.phi = self.phi[:,1:]
        
        # Construct local views in the ambient space
        # and obtain radius of each view
        self.U, epsilon = local_views_in_ambient_space(self.d_e, self.k)
        
        # Compute Atilde
        self.Atilde = compute_Atilde(self.phi, self.d_e, self.U, epsilon, self.p, self.d)
        
        # Compute gamma
        self.gamma = np.sqrt(1/(np.dot(self.U,self.phi**2)/np.sum(self.U,1)[:,np.newaxis]))
        
        print('\nConstructing low distortion local views...')
        
        # Compute LDLE: Low Distortion Local Eigenmaps
        self.Psi_gamma0, self.Psi_i0, self.zeta0 = self.compute_LDLE()
        
        # Postprocess LDLE
        self.Psi_gamma, self.Psi_i, self.zeta = self.postprocess_LDLE()
        
        # Compute beta
        self.beta = self.compute_beta()
            
        print('\nClustering to obtain low distortion intermediate views...')
            
        # Clustering to obtain intermediate views
        self.c, self.n_C, self.Utilde, self.Psitilde_i, self.Psitilde_gamma,\
            self.betatilde, self.zetatilde = self.construct_intermediate_views()
        
        # Compute |Utilde_{mm'}|
        self.n_Utilde_Utilde = np.dot(self.Utilde, self.Utilde.T)
        np.fill_diagonal(self.n_Utilde_Utilde, 0)
        
        print('\nInitializing parameters and computing initial global embedding...')
        
        # Compute initial global embedding
        self.T_init, self.v_init, self.y_init,\
        self.s_0, self.color_of_pts_on_tear_init = self.compute_initial_global_embedding()
        
        print('\nRefining parameters and computing final global embedding...')
        
        # Compute final global embedding
        if not self.use_geotorch:
            print('Using GPA...')
            self.T_final, self.v_final, self.y_final,\
            self.color_of_pts_on_tear_final = self.compute_final_global_embedding()
        else:
            print('Using geotorch...')
            self.T_final, self.v_final, self.y_final,\
            self.color_of_pts_on_tear_final = self.compute_final_global_embedding_geotorch_based()
    
    def compute_LDLE(self, print_prop = 0.25):
        # initializations
        n, N = self.phi.shape
        N = self.phi.shape[1]
        d = self.d
        tau = self.tau
        delta = self.delta
        
        print_freq = np.int(n*print_prop)
        
        Psi_gamma = np.zeros((n,d))
        Psi_i = np.zeros((n,d),dtype='int')
        zeta = np.zeros(n)

        # iterate over points in the data
        for k in range(n):
            if print_freq and np.mod(k, print_freq)==0:
                print('Psi,zeta: %d points processed...' % k)
            
            # to store i_1, ..., i_d
            i = np.zeros(d, dtype='int')
            
            # Grab the precomputed U_k, Atilde_{kij}, gamma_{ki}
            U_k = self.U[k,:]==1
            Atilde_k = self.Atilde[k,:,:]
            gamma_k = self.gamma[k,:]
            
            # Compute theta_1
            Atikde_kii = Atilde_k.diagonal()
            theta_1 = np.percentile(Atikde_kii, tau)
            
            # Compute Stilde_k
            Stilde_k = Atikde_kii >= theta_1
            
            # Compute i_1
            r_1 = np.argmax(Stilde_k)
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
                
                theta_s=np.max([theta_s,np.min([np.max(temp_),1e-4])])
                
                # Compute i_s
                r_s = np.argmax((Hs_kii>=theta_s) & Stilde_k)
                Hs_kir_s = Atilde_k[:,[r_s]] - np.dot(Atilde_k[:,i_prev], np.dot(temp, Atilde_k[i_prev,r_s][:,np.newaxis]))
                temp = gamma_k * np.abs(Hs_kir_s.flatten())
                alpha_s = np.max(temp * Stilde_k)
                i[s]=np.argmax((temp >= delta*alpha_s) & Stilde_k);
            
            # Compute Psi_k
            Psi_gamma[k,:] = gamma_k[i]
            Psi_i[k,:] = i
            
            # Compute zeta_{kk}
            zeta[k] = compute_zeta(self.d_e[np.ix_(U_k,U_k)], eval_param(self.phi, Psi_gamma, Psi_i, k, U_k))
            
        print('Psi,zeta: all %d points processed...' % n)
        return Psi_gamma, Psi_i, zeta
    
    def postprocess_LDLE(self):
        # initializations
        n = self.d_e.shape[0]
        Psi_i = np.copy(self.Psi_i0)
        Psi_gamma = np.copy(self.Psi_gamma0)
        zeta = np.copy(self.zeta0)
        
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
                    Psi_kp_on_U_k = eval_param(self.phi, Psi_gamma, Psi_i, kp, U_k)
                    zeta_kkp = compute_zeta(self.d_e[np.ix_(U_k,U_k)], Psi_kp_on_U_k)
                    
                    # if zeta_{kk'} < zeta_{kk}
                    if zeta_kkp < zeta[k]:
                        zeta[k] = zeta_kkp
                        new_param_of[k] = kp
                        param_changed_new[k] = 1
                        
            Psi_i = Psi_i[new_param_of,:]
            Psi_gamma = Psi_gamma[new_param_of,:]
            param_changed_old = param_changed_new
            N_replaced = np.sum(param_changed_new)
            
            print("After iter %d, max distortion is %f" % (itr, np.max(zeta)))
            itr = itr + 1
        
        return Psi_gamma, Psi_i, zeta
    
    def compute_beta(self):
        n = self.phi.shape[0]
        beta = np.zeros(n)
        for k in range(n):
            U_k = self.U[k,:]==1
            d_e_U_k = self.d_e[np.ix_(U_k,U_k)]
            if d_e_U_k.shape[0]==1:
                self.beta[k] = 1
            else:
                Psi_k_on_U_k = eval_param(self.phi, self.Psi_gamma, self.Psi_i, k, U_k)
                beta[k]=np.median(squareform(d_e_U_k))/np.median(pdist(Psi_k_on_U_k))
        return beta
    
    def construct_intermediate_views(self):
        # initializations
        n, N = self.phi.shape
        c = np.arange(n)
        n_C = np.zeros(n) + 1
        Utilde = np.copy(self.U)
        
        # Vary eta from 2 to eta_{min}
        for eta in range(2,self.eta_min+1):
            print('#nodes in views with sz < %d = %d' % (eta, np.sum(n_C[c]<eta)));
            
            # Compute cost_k and d_k (dest_k) for all k
            cost = np.zeros(n)+np.inf
            dest = np.zeros(n,dtype='int')-1
            for k in range(n):
                cost[k], dest[k] = cost_of_moving(k, self.d_e, self.U, self.phi, self.Psi_gamma,
                                                   self.Psi_i, c, n_C, Utilde, eta, self.eta_max)
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
                    cost[k], dest[k] = cost_of_moving(k, self.d_e, self.U, self.phi, self.Psi_gamma,
                                                       self.Psi_i, c, n_C, Utilde, eta, self.eta_max)
                
                # Recompute point with minimum cost
                # Recompute k and cost^*
                k = np.argmin(cost)
                cost_star = cost[k]
            print('Remaining #nodes in views with sz < %d = %d' % (eta, np.sum(n_C[c]<eta)))
        
        # Prune empty clusters
        non_empty_C = n_C > 0
        M = np.sum(non_empty_C)
        old_to_new_map = np.arange(n)
        old_to_new_map[non_empty_C] = np.arange(M)
        c = old_to_new_map[c]
        n_C = n_C[non_empty_C]
        
        # Compute Psitilde, betatilde
        Psitilde_i = self.Psi_i[non_empty_C,:]
        Psitilde_gamma = self.Psi_gamma[non_empty_C,:]
        betatilde = self.beta[non_empty_C]
        
        # Compute Utilde_m
        Utilde = np.zeros((M,n),dtype=bool)
        for m in range(M):
            Utilde[m,:] = np.any(self.U[c==m,:], 0)
        
        self.Utilde = Utilde
        
        # Compute zetatilde
        zetatilde = np.zeros(M);
        for m in range(M):
            Utilde_m = Utilde[m,:]
            zetatilde[m] = compute_zeta(self.d_e[np.ix_(Utilde_m,Utilde_m)],
                                        eval_param(self.phi, Psitilde_gamma,
                                                   Psitilde_i, m, Utilde_m))
        
        print("After clustering, max distortion is %f" % (np.max(zetatilde)))
        return c, n_C, Utilde, Psitilde_i, Psitilde_gamma, betatilde, zetatilde
    
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
                print('Ambiguous overlaps checked for %d points' % m)
            for mp in np.where(n_Utilde_Utilde[m,:] > 0)[0].tolist():
                if mp > m:
                    # Compute Utilde_{mm'}
                    Utilde_mmp = self.Utilde[m,:]*self.Utilde[mp,:]
                    # Compute V_{mm'}, V_{m'm}, Vbar_{mm'}, Vbar_{m'm}
                    V_mmp = eval_param(self.phi, self.Psitilde_gamma,
                                       self.Psitilde_i, m, Utilde_mmp, self.betatilde)
                    V_mpm = eval_param(self.phi, self.Psitilde_gamma,
                                       self.Psitilde_i, mp, Utilde_mmp, self.betatilde)
                    Vbar_mmp = V_mmp - np.mean(V_mmp,0)[np.newaxis,:]
                    Vbar_mpm = V_mpm - np.mean(V_mpm,0)[np.newaxis,:]
                    # Compute ambiguity as the minimum singular value of
                    # the d x d matrix Vbar_{mm'}^TVbar_{m'm}
                    W[m,mp] = svdvals(np.dot(Vbar_mmp.T,Vbar_mpm))[-1]
                    W[mp,m] = W[m,mp]

        print('Ambiguous overlaps checked for %d points' % M)
        # Compute maximum spanning tree of W
        T = minimum_spanning_tree(coo_matrix(-W))
        # Compute breadth first order in T starting from s_1
        s, rho = breadth_first_order(T, s_1, directed=False) #(ignores edge weights)
        print('Seq of intermediate views and their predecessors computed.')
        if s.shape[0] < M:
            raise RuntimeError('Multiple connected components detected')
        return s, rho
    
    def compute_Utildeg(self, y):
        M,n = self.Utilde.shape
        d_e_ = squareform(pdist(y))
        Ug, _ = local_views_in_ambient_space(d_e_, np.min([self.k * self.nu, d_e_.shape[0]-1]))
        Utildeg = np.zeros((M,n),dtype=bool)
        for m in range(M):
            Utildeg[m,:] = np.any(Ug[self.c==m,:], 0)
            
        # |Utildeg_{mm'}|
        n_Utildeg_Utildeg = np.dot(Utildeg, Utildeg.T)
        np.fill_diagonal(n_Utildeg_Utildeg, 0)
            
        return Utildeg, n_Utildeg_Utildeg
       
    # Computes Z_s for the case when to_tear is True.
    # Input Z_s is the Z_s for the case when to_tear is False.
    # Output Z_s is a subset of input Z_s.
    def compute_Z_s_to_tear(self, y, s, Z_s):
        n_Z_s = Z_s.shape[0]
        C_s_U_C_Z_s = (self.c == s) | np.isin(self.c, Z_s)
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

        # track the next color to assign
        cur_color = 1

        # Iterate over views
        for m in range(M):
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
                    temp_m = temp[i,:] & (self.c==m)
                    temp_mp = temp[i,:] & (self.c==mp)
                    if np.any(temp_m) and np.any(temp_mp):
                        color_of_pts_on_tear[temp_m|temp_mp] = cur_color
                        cur_color += 1
                        
        return color_of_pts_on_tear

    def compute_initial_global_embedding(self, print_prop = 0.25):
        # Intializations
        M = self.Utilde.shape[0]
        n = self.phi.shape[0]
        d = self.d
        print_freq = int(M*print_prop)

        T = np.tile(np.eye(d),[M,1,1])
        v = np.zeros((M,d))
        err = np.zeros(M)
        y = np.zeros((n,d))

        # Compute the sequence s in which intermediate views
        # are visited. rho[m] = predecessor of mth view.
        seq, rho = self.compute_seq_of_intermediate_views()

        # Boolean array to keep track of already visited views
        is_visited = np.zeros(M, dtype=bool)
        is_visited[seq[0]] = True

        C_s_0 = self.c==seq[0]
        y[C_s_0,:] = eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                               seq[0], C_s_0, self.betatilde, T, v)

        # Traverse views from 2nd view
        for m in range(1,M):
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
                V_s_p = eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                                   s, Utilde_s_p, self.betatilde, T, v)
                V_p_s = eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                                   p, Utilde_s_p, self.betatilde, T, v)
                T[s,:,:], v[s,:], err[s] = procrustes(V_s_p, V_p_s)

                # Compute temporary global embedding of point in sth cluster
                C_s = self.c==s
                y[C_s,:] = eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                                       s, C_s, self.betatilde, T, v)

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
                mu_s[Utilde_s_mp,:] += eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                                                  mp, Utilde_s_mp, self.betatilde, T, v)

            # Compute T_s and v_s by aligning the embedding of the overlap
            # between sth view and the views in Z_s, with the centroid mu_s
            temp = n_Utilde_s_Z_s > 0
            mu_s = mu_s[temp,:] / n_Utilde_s_Z_s[temp,np.newaxis]
            V_s_Z_s = eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                                 s, temp, self.betatilde, T, v)
            
            T_s, v_s, err[s] = procrustes(V_s_Z_s, mu_s)
            
            # Update T_s, v_
            T[s,:,:] = np.dot(T[s,:,:], T_s)
            v[s,:] = np.dot(v[s,:][np.newaxis,:], T_s) + v_s

            # Mark sth view as visited
            is_visited[s] = True

            # Compute global embedding of point in sth cluster
            C_s = self.c==s
            y[C_s,:] = eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                                   s, C_s, self.betatilde, T, v)

        print('error:', np.mean(err))
        
        if self.to_tear:
            color_of_pts_on_tear = self.compute_color_of_pts_on_tear(y)
        else:
            color_of_pts_on_tear = None
        
        if self.vis is not None:
            v_opts = self.vis_y_options
            self.vis.global_embedding(y, v_opts['labels'], v_opts['cmap0'],
                                      color_of_pts_on_tear, v_opts['cmap1'],
                                      'Initial')
            plt.waitforbuttonpress(1)
        
        return T, v, y, seq[0], color_of_pts_on_tear
    
    def compute_final_global_embedding(self):
        # Intializations
        M = self.Utilde.shape[0]
        n = self.phi.shape[0]
        d = self.d

        T = np.copy(self.T_init)
        v = np.copy(self.v_init)

        np.random.seed(42) # for reproducbility

        y = np.copy(self.y_init)
        
        # If to tear the closed manifolds
        if self.to_tear:
            # Compute |Utildeg_{mm'}|
            _, n_Utildeg_Utildeg = self.compute_Utildeg(y)

        # Refine global embedding y
        for it0 in range(self.max_iter0):
            print('Iteration: %d' % it0)

            err = np.zeros(M)
            
            # Traverse over intermediate views in a random order
            seq = np.random.permutation(M)

            # For a given seq, refine the global embedding
            for it1 in range(self.max_iter1):
                for s in seq.tolist():
                    # Never refine s_0th intermediate view
                    if s == self.s_0:
                        C_s = self.c==s
                        y[C_s,:] = eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                                           s, C_s, self.betatilde, T, v)
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

                    # Compute centroid mu_s
                    # n_Utilde_s_Z_s[k] = #views in Z_s which contain
                    # kth point if kth point is in the sth view, else zero
                    n_Utilde_s_Z_s = np.zeros(n, dtype=int)
                    mu_s = np.zeros((n,d))
                    for mp in Z_s:
                        Utilde_s_mp = Utilde_s & self.Utilde[mp,:]
                        n_Utilde_s_Z_s[Utilde_s_mp] += 1
                        mu_s[Utilde_s_mp,:] += eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                                                          mp, Utilde_s_mp, self.betatilde, T, v)

                    temp = n_Utilde_s_Z_s > 0
                    mu_s = mu_s[temp,:] / n_Utilde_s_Z_s[temp,np.newaxis]

                    # Compute T_s and v_s by aligning the embedding of the overlap
                    # between sth view and the views in Z_s, with the centroid mu_s
                    V_s_Z_s = eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                                         s, temp, self.betatilde, T, v)
                    T_s, v_s, err[s] = procrustes(V_s_Z_s, mu_s)

                    # Update T_s, v_s
                    T[s,:,:] = np.dot(T[s,:,:], T_s)
                    v[s,:] = np.dot(v[s,:][np.newaxis,:], T_s) + v_s

                    # Compute global embedding of points in sth cluster
                    C_s = self.c==s
                    y[C_s,:] = eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                                           s, C_s, self.betatilde, T, v)

            # If to tear the closed manifolds
            if self.to_tear:
                # Compute |Utildeg_{mm'}|
                _, n_Utildeg_Utildeg = self.compute_Utildeg(y)
                color_of_pts_on_tear = self.compute_color_of_pts_on_tear(y, n_Utildeg_Utildeg)
            else:
                color_of_pts_on_tear = None
            
            print('error:', np.mean(err))
            # Visualize current embedding
            if self.vis is not None:
                v_opts = self.vis_y_options
                self.vis.global_embedding(y, v_opts['labels'], v_opts['cmap0'],
                                          color_of_pts_on_tear, v_opts['cmap1'],
                                          'Iter: %d' % it0)
                plt.waitforbuttonpress(1)

        return T, v, y, color_of_pts_on_tear
    
    def compute_final_global_embedding_geotorch_based(self):
        # Intializations
        M = self.Utilde.shape[0]
        n = self.phi.shape[0]
        d = self.d

        T = np.copy(self.T_init)
        v = np.copy(self.v_init)

        # Initialize pytorch parameters
        Tv = [] # T_m, v_m = Tv[m].weight, Tv[m].bias
        params = []
        for m in range(M):
            Tv.append(torch.nn.Linear(d, d, bias=True).double())
            # Orthogonality constraints on weights
            geotorch.orthogonal(Tv[-1], "weight")
            # Initialize weights
            Tv[-1].weight = torch.from_numpy(T[m,:,:].T)
            Tv[-1].bias.data = torch.from_numpy(v[m,:])
            if m != self.s_0: # Never update parameters of first intermediate view
                params += Tv[-1].parameters()

        optim = torch.optim.Adam(params, lr=self.geotorch_options['lr'])

        np.random.seed(42) # for reproducbility

        y = np.copy(self.y_init)

        # If to tear the closed manifolds
        if self.to_tear:
            # Compute |Utildeg_{mm'}|
            _, n_Utildeg_Utildeg = self.compute_Utildeg(y)

        # Refine global embedding y
        for it0 in range(self.max_iter0):
            print('Iteration: %d' % it0)
            for epoch in range(self.max_iter1):
                loss = torch.tensor(0)
                for s in range(M):
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

                    for mp in Z_s:
                        Utilde_s_mp = Utilde_s & self.Utilde[mp,:]
                        V_s_mp = eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                                              s, Utilde_s_mp, self.betatilde)
                        V_mp_s = eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                                              mp, Utilde_s_mp, self.betatilde)
                        #pdb.set_trace()
                        loss = loss + torch.sum(torch.square(Tv[s](torch.from_numpy(V_s_mp)) - Tv[mp](torch.from_numpy(V_mp_s))))

                optim.zero_grad()
                loss.backward()
                optim.step()
                print("Epoch {} Loss: {:.6f}".format(epoch, loss))

            for s in range(M):
                # Store learned T and v
                T[s,:,:] = Tv[s].weight.detach().numpy().T
                v[s,:] = Tv[s].bias.detach().numpy()
                # Compute global embedding of points in sth cluster
                C_s = self.c==s
                y[C_s,:] = eval_param(self.phi, self.Psitilde_gamma, self.Psitilde_i,
                                       s, C_s, self.betatilde, T, v)

            # If to tear the closed manifolds
            if self.to_tear:
                # Compute |Utildeg_{mm'}|
                _, n_Utildeg_Utildeg = self.compute_Utildeg(y)
                color_of_pts_on_tear = self.compute_color_of_pts_on_tear(y, n_Utildeg_Utildeg)
            else:
                color_of_pts_on_tear = None

            # Visualize current embedding
            if self.vis is not None:
                v_opts = self.vis_y_options
                self.vis.global_embedding(y, v_opts['labels'], v_opts['cmap0'],
                                          color_of_pts_on_tear, v_opts['cmap1'],
                                          'Iter: %d' % it0)
                plt.waitforbuttonpress(1)

        return T, v, y, color_of_pts_on_tear