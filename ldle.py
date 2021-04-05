import numpy as np
from numpy.linalg import inv

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigs, eigsh
from scipy.stats.distributions import chi2

from sklearn.neighbors import NearestNeighbors

def eval_param(phi, Psi_gamma, Psi_i, k, mask, beta=None):
    if beta is None:
        return Psi_gamma[k,:][np.newaxis,:] * phi[np.ix_(mask,Psi_i[k,:])]
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
    elif gl_type == 'random_walk':
        L, D = laplacian(K, normed=False,
                         return_diag=True,
                         use_out_degree=use_out_degree)
        L.data /= D[L.row]
        if return_diag:
            return L, D
        else:
            return L
    

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
            print('Atilde_k: %d points processed...' % k)
        U_k = U[k,:]==1
        dphi_k = phi[U_k,:]-phi[k,:]
        Atilde[k,:,:] = np.dot(dphi_k.T, dphi_k*(Gtilde[k,U_k][:,np.newaxis]))
    
    print('Atilde_k: all points processed...')
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
                 eta_max = 100):
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
        
        # Construct graph Laplacian
        self.L = graph_laplacian(self.d_e, self.k_nn,
                                 self.k_tune, self.gl_type)
        
        # Eigendecomposition of graph Laplacian
        # Note: Eigenvalues are returned sorted.
        # Following is needed for reproducibility of lmbda and phi
        np.random.seed(2)
        v0 = np.random.uniform(0,1,self.L.shape[0])
        if self.gl_type != 'random_walk':
            self.lmbda, self.phi = eigsh(self.L, k=self.N+1, v0=v0, which='SM')
        else:
            self.lmbda, self.phi = eigs(self.L, k=self.N+1, v0=v0, which='SM')
        
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
        
        # Compute LDLE: Low Distortion Local Eigenmaps
        self.Psi_gamma0, self.Psi_i0, self.zeta0 = self.compute_LDLE()
        
        # Postprocess LDLE
        self.Psi_gamma, self.Psi_i, self.zeta = self.postprocess_LDLE()
        
        # Compute beta
        self.beta = self.compute_beta()
            
        # Clustering to obtain intermediate views
        self.c, self.Utilde, self.Psitilde_i, self.Psitilde_gamma,\
        self.betatilde, self.zetatilde = self.construct_intermediate_views()
        
    
    
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
                i_prev = i[0:s];
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
        Psi_i = self.Psi_i0
        Psi_gamma = self.Psi_gamma0
        zeta = self.zeta0
        
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
        Utilde = np.zeros((M,n))
        for m in range(M):
            Utilde[m,:] = np.any(self.U[c==m,:], 0)
        
        Utilde = Utilde==1
        self.Utilde = Utilde
        
        # Compute zetatilde
        zetatilde = np.zeros(M);
        for m in range(M):
            Utilde_m = Utilde[m,:]
            zetatilde[m] = compute_zeta(self.d_e[np.ix_(Utilde_m,Utilde_m)],
                                        eval_param(self.phi, Psitilde_gamma,
                                                   Psitilde_i, m, Utilde_m))
        
        print("After clustering, max distortion is %f" % (np.max(zetatilde)))
        return c, Utilde, Psitilde_i, Psitilde_gamma, betatilde, zetatilde