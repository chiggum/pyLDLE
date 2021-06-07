import sys
import os

import numpy as np

from matplotlib import pyplot as plt
plt.rcParams.update({'scatter.marker':'.'})
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data

def eval_param(phi, Psi_gamma, Psi_i, k, mask, beta=None, T=None, v=None):
    if beta is None:
        return Psi_gamma[k,:][np.newaxis,:] * phi[np.ix_(mask,Psi_i[k,:])]
    else:
        if T is not None and v is not None:
            return np.dot(beta[k]*Psi_gamma[k,:][np.newaxis,:] * phi[np.ix_(mask,Psi_i[k,:])], T[k,:,:]) + v[[k],:]
        else:
            return beta[k]*Psi_gamma[k,:][np.newaxis,:] * phi[np.ix_(mask,Psi_i[k,:])]

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def on_close(event):
    raise RuntimeError("Figure closed.")

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

class Visualize:
    def __init__(self, save_dir=''):
        self.save_dir = save_dir
        if self.save_dir:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
        pass
    
    def data(self, X, labels, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], s=s, c=labels, cmap='jet')
            plt.axis('image')
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            ax.autoscale()
            ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=labels, cmap='jet')
        plt.title('Data')
        if self.save_dir:
            plt.savefig(self.save_dir+'/data.png') 
        
    def eigenvalues(self, lmbda, figsize=None):
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.plot(lmbda, 'o-')
        plt.ylabel('$\lambda_i$')
        plt.xlabel('i')
        plt.title('Eigenvalues')
        if self.save_dir:
            plt.savefig(self.save_dir+'/eigenvalues.png') 
        
    def gamma(self, X, gamma, i, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], s=s, c=gamma[:,i], cmap='jet')
            plt.axis('image')
            plt.colorbar()
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            ax.autoscale()
            #ax.set_aspect('equal')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=gamma[:,i], cmap='jet')
            fig.colorbar(p)
        plt.title('$\gamma_{%d}$'%i)
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/gamma'):
                os.makedirs(self.save_dir+'/gamma')
            plt.savefig(self.save_dir+'/gamma/'+str(i)+'.png') 
    
    def eigenvector(self, X, phi, i, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], s=s, c=phi[:,i], cmap='jet')
            plt.axis('image')
            plt.colorbar()
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            ax.autoscale()
            #ax.set_aspect('equal')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=phi[:,i], cmap='jet')
            fig.colorbar(p)
        plt.title('$\phi_{%d}$'%i)
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/eigvecs'):
                os.makedirs(self.save_dir+'/eigvecs')
            plt.savefig(self.save_dir+'/eigvecs/'+str(i)+'.png') 
    
    def grad_phi(self, X, phi, grad_phi, i, prop=0.01, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        n = X.shape[0]
        np.random.seed(42)
        
        mask = np.random.uniform(0,1,n)<prop
        #mod = int(n*prop)
        #mask = np.mod(np.arange(n),mod) == 0
        
        if X.shape[1] == 2:
            plt.subplot(121)
            plt.scatter(X[:,0], X[:,1], s=s, c=phi[:,i], cmap='jet')
            plt.axis('image')
            #plt.colorbar()
            plt.title('$\phi_{%d}$'%i)
            plt.subplot(122)
            plt.quiver(X[mask,0], X[mask,1],grad_phi[mask,i,0], grad_phi[mask,i,1])
            plt.axis('image')
            plt.title('$\\nabla\\phi_{%d}$'%i)
        elif X.shape[1] == 3:
            ax = fig.add_subplot(121,projection='3d')
            ax.autoscale()
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=phi[:,i], cmap='jet')
            fig.colorbar(p, ax=ax)
            plt.title('$\phi_{%d}$'%i)
            ax = fig.add_subplot(122,projection='3d')
            ax.autoscale()
            p = ax.quiver(X[mask,0], X[mask,1], X[mask,2], grad_phi[mask,i,0], grad_phi[mask,i,1], grad_phi[mask,i,2])
            plt.title('$\\nabla\phi_{%d}$'%i)
        
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/grad_phi'):
                os.makedirs(self.save_dir+'/grad_phi')
            plt.savefig(self.save_dir+'/grad_phi/'+str(i)+'.png') 
    
    def Atilde(self, X, phi, i, j, Atilde, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.subplot(131)
            plt.scatter(X[:,0], X[:,1], s=s, c=phi[:,i], cmap='jet')
            plt.axis('image')
            plt.colorbar()
            plt.title('$\phi_{%d}$'%i)
            plt.subplot(132)
            plt.scatter(X[:,0], X[:,1], s=s, c=phi[:,j], cmap='jet')
            plt.axis('image')
            plt.colorbar()
            plt.title('$\phi_{%d}$'%j)
            plt.subplot(133)
            plt.scatter(X[:,0], X[:,1], s=s, c=Atilde[:,i,j], cmap='jet')
            plt.axis('image')
            plt.colorbar()
            plt.title('$\widetilde{A}_{:%d%d}$'%(i,j))
        elif X.shape[1] == 3:
            ax = fig.add_subplot(131,projection='3d')
            ax.autoscale()
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=phi[:,i], cmap='jet')
            fig.colorbar(p, ax=ax)
            plt.title('$\phi_{%d}$'%i)
            ax = fig.add_subplot(132,projection='3d')
            ax.autoscale()
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=phi[:,i], cmap='jet')
            fig.colorbar(p, ax=ax)
            plt.title('$\phi_{%d}$'%j)
            ax = fig.add_subplot(133,projection='3d')
            ax.autoscale()
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=Atilde[:,i,j], cmap='jet')
            fig.colorbar(p, ax=ax)
            plt.title('$\widetilde{A}_{:%d%d}$'%(i,j))
        
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/Atilde'):
                os.makedirs(self.save_dir+'/Atilde')
            plt.savefig(self.save_dir+'/Atilde/'+str(i)+'_'+str(j)+'.png') 
    
    def n_eigvecs_w_grad_lt(self, X, Atilde, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        
        if X.shape[1] == 3:
            ax = fig.add_subplot(122,projection='3d')
            cb = None
        
        Atilde_diag = np.diagonal(Atilde, axis1=1, axis2=2)
        
        prctiles = np.arange(100)
        plt.subplot(121)
        plt.plot(prctiles, np.percentile(Atilde_diag.flatten(), prctiles), 'bo-')
        plt.xlabel('percentiles')
        plt.title('$\\widetilde{A}_{kii}$\nDouble click = Select threshold\nPress button = quit')
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        while True:
            plt.subplot(121)
            
            to_exit = plt.waitforbuttonpress(timeout=20)
            if to_exit is None:
                print('Timed out')
                break
                
            if to_exit:
                plt.close()
                return
            
            thresh = plt.ginput(1)
            thresh = thresh[0][1]
            
            plt.cla()
            
            plt.plot(prctiles, np.percentile(Atilde_diag.flatten(), prctiles), 'bo-')
            plt.plot([0,100], [thresh]*2, 'r-')
            plt.xlabel('percentiles')
            plt.title('$\\widetilde{A}_{kii}$, threshold = %f\nDouble click = Select threshold\nPress button = quit' % thresh)
            fig.canvas.draw()
            fig.canvas.flush_events()
        
            n_grad_lt = np.sum(Atilde_diag < thresh, 1)
        
            if X.shape[1] == 2:
                plt.subplot(122)
                plt.cla()
                plt.scatter(X[:,0], X[:,1], s=s, c=n_grad_lt, cmap='jet')
                plt.axis('image')
                plt.colorbar()
                plt.title('$n_k = \sum_{i}\widetilde{A}_{kii} < %f$'% thresh)
            elif X.shape[1] == 3:
                ax.cla()
                ax.autoscale()
                p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=n_grad_lt, cmap='jet')
                if cb is not None:
                    cb.remove()
                cb = fig.colorbar(p, ax=ax)
                ax.set_title('$n_k = \sum_{i}\widetilde{A}_{kii} < %f$'% thresh)
                
            if self.save_dir:
                if not os.path.isdir(self.save_dir+'/n_eigvecs_w_grad_lt'):
                    os.makedirs(self.save_dir+'/n_eigvecs_w_grad_lt')
                plt.savefig(self.save_dir+'/n_eigvecs_w_grad_lt/'+str(thresh)+'.png') 
    
    def distortion(self, X, zeta, title, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], s=s, c=zeta, cmap='jet')
            plt.axis('image')
            plt.colorbar()
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            ax.autoscale()
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=zeta, cmap='jet')
            fig.colorbar(p)
        plt.title(title)
        if self.save_dir:
            plt.savefig(self.save_dir+'/'+title+'.png') 
            
    def dX(self, X, ddX, title, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.subplot(121)
            plt.scatter(X[:,0], X[:,1], s=s, c=ddX, cmap='jet')
            plt.colorbar()
            plt.title('distance from dX')
            plt.subplot(122)
            plt.scatter(X[:,0], X[:,1], s=s, c=ddX==0, cmap='jet')
            plt.colorbar()
            plt.title('dX')
        elif X.shape[1] == 3:
            ax = fig.add_subplot(121, projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=ddX, cmap='jet')
            fig.colorbar(p, ax=ax)
            ax.set_title('distance from dX')
            
            ax = fig.add_subplot(122, projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=ddX==0, cmap='jet')
            fig.colorbar(p, ax=ax)
            ax.set_title('dX')
            
        if self.save_dir:
            plt.savefig(self.save_dir+'/'+title+'.png') 
    
    def intrinsic_dim(self, X, chi, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        if X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], s=s, c=chi, cmap='jet')
            plt.axis('image')
            plt.colorbar()
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            ax.autoscale()
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=chi, cmap='jet')
            fig.colorbar(p)
        plt.title('\chi')
    
    def chosen_eigevec_inds_for_local_views(self, X, Psi_i, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.subplot(211)
            plt.scatter(X[:,0], X[:,1], s=s, c=Psi_i[:,0], cmap='jet')
            plt.axis('image')
            plt.colorbar()
            plt.title('\\phi_{i_1}')
            plt.subplot(212)
            plt.scatter(X[:,0], X[:,1], s=s, c=Psi_i[:,1], cmap='jet')
            plt.axis('image')
            plt.colorbar()
            plt.title('\\phi_{i_2}')
        elif X.shape[1] == 3:
            ax = fig.add_subplot(211, projection='3d')
            ax.autoscale()
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=Psi_i[:,0], cmap='jet')
            fig.colorbar(p)
            ax.set_title('\\phi_{i_1}')
            ax = fig.add_subplot(212, projection='3d')
            ax.autoscale()
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=Psi_i[:,1], cmap='jet')
            fig.colorbar(p)
            ax.set_title('\\phi_{i_2}')
        
        if self.save_dir:
            plt.savefig(self.save_dir+'/chosen_eigvecs_for_local_views.png') 
    
    def chosen_eigevec_inds_for_intermediate_views(self, X, Psitilde_i, c, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.subplot(211)
            plt.scatter(X[:,0], X[:,1], s=s, c=Psitilde_i[c,0], cmap='jet')
            plt.axis('image')
            plt.colorbar()
            plt.title('\\phi_{i_1}')
            plt.subplot(212)
            plt.scatter(X[:,0], X[:,1], s=s, c=Psitilde_i[c,1], cmap='jet')
            plt.axis('image')
            plt.colorbar()
            plt.title('\\phi_{i_2}')
        elif X.shape[1] == 3:
            ax = fig.add_subplot(211, projection='3d')
            ax.autoscale()
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=Psitilde_i[c,0], cmap='jet')
            fig.colorbar(p)
            plt.title('\\phi_{i_1}')
            ax = fig.add_subplot(212, projection='3d')
            ax.autoscale()
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=Psitilde_i[c,1], cmap='jet')
            fig.colorbar(p)
            ax.set_title('\\phi_{i_2}')
        
        if self.save_dir:
            plt.savefig(self.save_dir+'/chosen_eigvecs_for_intermediate_views.png') 
    
    def local_views(self, X, phi, U, gamma, Atilde, Psi_gamma, Psi_i, zeta, k=None, save_subdir='', figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        is_3d_data = X.shape[1] == 3
        n,N = phi.shape
        
        fig = plt.figure(1, figsize=figsize)
        fig.canvas.mpl_connect('close_event', on_close)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        
        cb = [None, None, None]
        ax = []
        if is_3d_data:
            for i in range(3):
                ax.append(fig.add_subplot(231+i, projection='3d'))
            for i in range(3,6):
                ax.append(fig.add_subplot(231+i))
                                        
            p = ax[0].scatter(X[:,0], X[:,1], X[:,2], s=s, c=zeta, cmap='jet')
        else:
            for i in range(6):
                ax.append(fig.add_subplot(231+i))
            
            p = ax[0].scatter(X[:,0], X[:,1], s=s, c=zeta, cmap='jet')
            ax[0].axis('image')
        
        ax[0].set_title('Double click = select a local view.\nPress button = exit.')
            
        cb[0] = plt.colorbar(p, ax=ax[0])
        cb[1] = plt.colorbar(p, ax=ax[1])
        cb[2] = plt.colorbar(p, ax=ax[2])
        
        k_not_available = k is None
        
        while True:
            plt.figure(1, figsize=figsize)
            if k_not_available:
                to_exit = plt.waitforbuttonpress(timeout=20)
                if to_exit:
                    plt.close()
                    return
                # Plot data with distortion colormap and the
                # selected local view in the ambient space
                ax[0]
                if is_3d_data:
                    plt.ginput(1)
                    k = np.random.randint(n)
                else:
                    X_k = plt.ginput(1)
                    X_k = np.array(X_k[0])[np.newaxis,:]
                    k = np.argmin(np.sum((X-X_k)**2,1))
                
            U_k = U[k,:]==1
            
            ax[0].cla()
            cb[0].remove()
            if is_3d_data:
                p = ax[0].scatter(X[:,0], X[:,1], X[:,2], s=s*(1-U_k), c=zeta, cmap='jet')
                ax[0].scatter(X[U_k,0], X[U_k,1], X[U_k,2], s=s, c='k')
            else:
                p = ax[0].scatter(X[:,0], X[:,1], s=s*(1-U_k), c=zeta, cmap='jet')
                ax[0].scatter(X[U_k,0], X[U_k,1], s=s, c='k')
                ax[0].axis('image')
            
            cb[0] = plt.colorbar(p, ax=ax[0])
            ax[0].set_title('$\\mathcal{M}$ and $U_{%d}$' % k)
            
            # Plot the corresponding local view in the embedding space
            y = eval_param(phi, Psi_gamma, Psi_i, k, np.ones(n)==1)
            ax[3]
            ax[3].cla()
            ax[3].scatter(y[:,0], y[:,1], s=s, c='r')
            ax[3].scatter(y[U_k,0], y[U_k,1], s=s, c='k')
            ax[3].axis('image')
            ax[3].set_title('$\\zeta_{%d%d}=%.3f\\'\
                              ' \\Phi_{%d}(\\mathcal{M})$ in red and $\\Phi_{%d}(U_{%d})$ in black'\
                              % (k, k, zeta[k], k, k, k))
            
            # Plot the chosen eigenvectors and scaled eigenvectors
            subplots = [232, 233]
            for j in range(len(subplots)):
                i_s = Psi_i[k,j]
                ax[j+1]
                ax[j+1].cla()
                cb[j+1].remove()
                if is_3d_data:
                    p = ax[j+1].scatter(X[:,0], X[:,1], X[:,2], s=s*(1-U_k), c=phi[:,i_s], cmap='jet')
                    ax[j+1].scatter(X[U_k,0], X[U_k,1], X[U_k,2], s=s, c='k')
                else:
                    p = ax[j+1].scatter(X[:,0], X[:,1], s=s*(1-U_k), c=phi[:,i_s], cmap='jet')
                    ax[j+1].scatter(X[U_k,0], X[U_k,1], s=s, c='k')
                    ax[j+1].axis('image')
                                        
                cb[j+1] = plt.colorbar(p, ax=ax[j+1])
                ax[j+1].set_title('$\\phi_{%d}$' % i_s)
            
            Atilde_k = np.abs(Atilde[k,:,:])
            Atilde_kii = np.sqrt(Atilde_k.diagonal()[:,np.newaxis])
            angles = (Atilde_k/Atilde_kii)/(Atilde_kii.T)
            
            prctiles = np.arange(100)
            ax[4]
            ax[4].cla()
            ax[4].plot(prctiles, np.percentile(angles.flatten(), prctiles), 'bo-')
            ax[4].plot([0,100], [0,0], 'g-')
            ax[4].plot([0,100], [angles[Psi_i[k,0],Psi_i[k,1]]]*2, 'r-')
            ax[4].set_xlabel('percentiles')
            ax[4].set_title('$|\\widetilde{A}_{%dij}|/(\\widetilde{A}_{%dii}\\widetilde{A}_{%djj})$' % (k, k, k))
            
            
            local_scales = gamma[[k],:].T*Atilde_kii
            #dlocal_scales = squareform(pdist(local_scales))
            dlocal_scales = np.log(local_scales/local_scales.T+1)
            
            ax[5]
            ax[5].cla()
            ax[5].plot(prctiles, np.percentile(dlocal_scales.flatten(), prctiles), 'bo-')
            ax[5].plot([0,100], [np.log(2)]*2, 'g-')
            ax[5].plot([0,100], [dlocal_scales[Psi_i[k,0],Psi_i[k,1]]]*2, 'r-')
            ax[5].set_xlabel('percentiles')
            ax[5].set_title('$\\log(\\gamma_{%di}\\sqrt{\\widetilde{A}_{%dii}} / \
                      \\gamma_{%dj}\\sqrt{\\widetilde{A}_{%djj}}+1)$' % (k, k, k, k))
                
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            if self.save_dir:
                if not os.path.isdir(self.save_dir+'/local_views/'+save_subdir):
                    os.makedirs(self.save_dir+'/local_views/'+save_subdir)
                plt.savefig(self.save_dir+'/local_views/'+save_subdir+'/'+str(k)+'.png')
            
            if not k_not_available:
                break
    
    def local_views_ltsap(self, X, phi, U, gamma, Atilde, Psi_gamma, Psi_i, zeta, save_subdir='', figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        is_3d_data = X.shape[1] == 3
        n,N = phi.shape
        
        fig = plt.figure(1, figsize=figsize)
        fig.canvas.mpl_connect('close_event', on_close)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        
        cb = [None, None, None]
        ax = []
        if is_3d_data:
            for i in range(3):
                ax.append(fig.add_subplot(231+i, projection='3d'))
            for i in range(3,6):
                ax.append(fig.add_subplot(231+i))
                                        
            p = ax[0].scatter(X[:,0], X[:,1], X[:,2], s=s, c=zeta, cmap='jet')
        else:
            for i in range(6):
                ax.append(fig.add_subplot(231+i))
            
            p = ax[0].scatter(X[:,0], X[:,1], s=s, c=zeta, cmap='jet')
            ax[0].axis('image')
        
        ax[0].set_title('Double click = select a local view.\nPress button = exit.')
            
        cb[0] = plt.colorbar(p, ax=ax[0])
        cb[1] = plt.colorbar(p, ax=ax[1])
        cb[2] = plt.colorbar(p, ax=ax[2])
        
        while True:
            plt.figure(1, figsize=figsize)
            to_exit = plt.waitforbuttonpress(timeout=20)
            if to_exit:
                plt.close()
                return
            # Plot data with distortion colormap and the
            # selected local view in the ambient space
            ax[0]
            if is_3d_data:
                plt.ginput(1)
                k = np.random.randint(n)
            else:
                X_k = plt.ginput(1)
                X_k = np.array(X_k[0])[np.newaxis,:]
                k = np.argmin(np.sum((X-X_k)**2,1))
                
            U_k = U[k,:]==1
            
            ax[0].cla()
            cb[0].remove()
            if is_3d_data:
                p = ax[0].scatter(X[:,0], X[:,1], X[:,2], s=s*(1-U_k), c=zeta, cmap='jet')
                ax[0].scatter(X[U_k,0], X[U_k,1], X[U_k,2], s=s, c='k')
            else:
                p = ax[0].scatter(X[:,0], X[:,1], s=s*(1-U_k), c=zeta, cmap='jet')
                ax[0].scatter(X[U_k,0], X[U_k,1], s=s, c='k')
                ax[0].axis('image')
            
            cb[0] = plt.colorbar(p, ax=ax[0])
            ax[0].set_title('$\\mathcal{M}$ and $U_{%d}$' % k)
            
            # Plot the corresponding local view in the embedding space
            y = eval_param(phi, Psi_gamma, Psi_i, k, np.ones(n)==1)
            ax[3]
            ax[3].cla()
            ax[3].scatter(y[:,0], y[:,1], s=s, c='r')
            ax[3].scatter(y[U_k,0], y[U_k,1], s=s, c='k')
            ax[3].axis('image')
            ax[3].set_title('$\\zeta_{%d%d}=%.3f\\'\
                              ' \\Phi_{%d}(\\mathcal{M})$ in red and $\\Phi_{%d}(U_{%d})$ in black'\
                              % (k, k, zeta[k], k, k, k))
            
            # Plot the chosen eigenvectors and scaled eigenvectors
            subplots = [232, 233]
            for j in range(len(subplots)):
                i_s = Psi_i[k,j]
                ax[j+1]
                ax[j+1].cla()
                cb[j+1].remove()
                if is_3d_data:
                    p = ax[j+1].scatter(X[:,0], X[:,1], X[:,2], s=s*(1-U_k), c=phi[:,i_s], cmap='jet')
                    ax[j+1].scatter(X[U_k,0], X[U_k,1], X[U_k,2], s=s, c='k')
                else:
                    p = ax[j+1].scatter(X[:,0], X[:,1], s=s*(1-U_k), c=phi[:,i_s], cmap='jet')
                    ax[j+1].scatter(X[U_k,0], X[U_k,1], s=s, c='k')
                    ax[j+1].axis('image')
                                        
                cb[j+1] = plt.colorbar(p, ax=ax[j+1])
                ax[j+1].set_title('$\\phi_{%d}$' % i_s)
            
            Atilde_k = np.abs(Atilde[k,:,:])
            Atilde_kii = np.sqrt(Atilde_k.diagonal()[:,np.newaxis])
            angles = (Atilde_k/Atilde_kii)/(Atilde_kii.T)
            
            prctiles = np.arange(100)
            ax[4]
            ax[4].cla()
            ax[4].plot(prctiles, np.percentile(angles.flatten(), prctiles), 'bo-')
            ax[4].plot([0,100], [0,0], 'g-')
            ax[4].plot([0,100], [angles[Psi_i[k,0],Psi_i[k,1]]]*2, 'r-')
            ax[4].set_xlabel('percentiles')
            ax[4].set_title('$|\\widetilde{A}_{%dij}|/(\\widetilde{A}_{%dii}\\widetilde{A}_{%djj})$' % (k, k, k))
            
            
            local_scales = gamma[[k],:].T*Atilde_kii
            #dlocal_scales = squareform(pdist(local_scales))
            dlocal_scales = np.log(local_scales/local_scales.T+1)
            
            ax[5]
            ax[5].cla()
            ax[5].plot(prctiles, np.percentile(dlocal_scales.flatten(), prctiles), 'bo-')
            ax[5].plot([0,100], [np.log(2)]*2, 'g-')
            ax[5].plot([0,100], [dlocal_scales[Psi_i[k,0],Psi_i[k,1]]]*2, 'r-')
            ax[5].set_xlabel('percentiles')
            ax[5].set_title('$\\log(\\gamma_{%di}\\sqrt{\\widetilde{A}_{%dii}} / \
                      \\gamma_{%dj}\\sqrt{\\widetilde{A}_{%djj}}+1)$' % (k, k, k, k))
                
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            if self.save_dir:
                if not os.path.isdir(self.save_dir+'/local_views/'+save_subdir):
                    os.makedirs(self.save_dir+'/local_views/'+save_subdir)
                plt.savefig(self.save_dir+'/local_views/'+save_subdir+'/'+str(k)+'.png')
    
    def intermediate_views(self, X, phi, Utilde, gamma, Atilde, Psitilde_gamma,
              Psitilde_i, zetatilde, c, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        is_3d_data = X.shape[1] == 3
        n,N = phi.shape

        zeta = zetatilde[c]
        
        fig = plt.figure(1, figsize=figsize)
        fig.canvas.mpl_connect('close_event', on_close)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        
        cb = [None, None, None]
        ax = []
        if is_3d_data:
            for i in range(3):
                ax.append(fig.add_subplot(231+i, projection='3d'))
            for i in range(3,6):
                ax.append(fig.add_subplot(231+i))
            p = ax[0].scatter(X[:,0], X[:,1], X[:,2], s=s, c=zeta, cmap='jet')
        else:
            for i in range(6):
                ax.append(fig.add_subplot(231+i))
            
            p = ax[0].scatter(X[:,0], X[:,1], s=s, c=zeta, cmap='jet')
            ax[0].axis('image')
        
        ax[0].set_title('Double click = select an intermediate view.\n Press button = exit.')
        
        cb[0] = plt.colorbar(p, ax=ax[0])
        cb[1] = plt.colorbar(p, ax=ax[1])
        cb[2] = plt.colorbar(p, ax=ax[2])
        
        while True:
            plt.figure(1, figsize=figsize)
            to_exit = plt.waitforbuttonpress(timeout=20)
            if to_exit is None:
                print('Timed out')
                break
                
            if to_exit:
                plt.close()
                return
            
            # Plot data with distortion colormap and the
            # selected local view in the ambient space
            ax[0]
            if is_3d_data:
                plt.ginput(1)
                k = np.random.randint(n)
            else:
                X_k = plt.ginput(1)
                X_k = np.array(X_k[0])[np.newaxis,:]
                k = np.argmin(np.sum((X-X_k)**2,1))

            m = c[k]
            Utilde_m = Utilde[m,:]
            
            ax[0].cla()
            cb[0].remove()
            if is_3d_data:
                p = ax[0].scatter(X[:,0], X[:,1], X[:,2], s=s*(1-Utilde_m), c=zeta, cmap='jet')
                ax[0].scatter(X[Utilde_m,0], X[Utilde_m,1], X[Utilde_m,2], s=s, c='k')
            else:
                ax[0].scatter(X[:,0], X[:,1], s=s*(1-Utilde_m), c=zeta, cmap='jet')
                ax[0].scatter(X[Utilde_m,0], X[Utilde_m,1], s=s, c='k')
                ax[0].axis('image')
            
            cb[0] = plt.colorbar(p, ax=ax[0])
            ax[0].set_title('$\\mathcal{M}$ and $\\widetilde{U}_{%d}$' % m)
            
            # Plot the corresponding local view in the embedding space
            y = eval_param(phi, Psitilde_gamma, Psitilde_i, m, np.ones(n)==1)
            ax[3]
            ax[3].cla()
            ax[3].scatter(y[:,0], y[:,1], s=s, c='r')
            ax[3].scatter(y[Utilde_m,0], y[Utilde_m,1], s=s, c='k')
            ax[3].axis('image')
            ax[3].set_title('$\\widetilde{\\zeta}_{%d%d}=%.3f\\'\
                          ' \\widetilde{\\Phi}_{%d}(\\mathcal{M})$ in red'\
                          ' and $\\widetilde{\\Phi}_{%d}(\\widetilde{U}_{%d})$ in black'\
                          % (m, m, zetatilde[m], m, m, m)) # zetatilde[m] == zeta[k]
            
            # Plot the chosen eigenvectors and scaled eigenvectors
            subplots = [232, 233]
            for j in range(len(subplots)):
                i_s = Psitilde_i[m,j]
                ax[j+1].cla()
                cb[j+1].remove()
                if is_3d_data:
                    p = ax[j+1].scatter(X[:,0], X[:,1], X[:,2], s=s*(1-Utilde_m), c=phi[:,i_s], cmap='jet')
                    ax[j+1].scatter(X[Utilde_m,0], X[Utilde_m,1], X[Utilde_m,2], s=s, c='k')
                else:
                    p = ax[j+1].scatter(X[:,0], X[:,1], s=s*(1-Utilde_m), c=phi[:,i_s], cmap='jet')
                    ax[j+1].scatter(X[Utilde_m,0], X[Utilde_m,1], s=s, c='k')
                    ax[j+1].axis('image')
                
                cb[j+1] = plt.colorbar(p, ax=ax[j+1])
                ax[j+1].set_title('$\\widetilde{\\phi}_{%d}$' % i_s)
            
            Atilde_k = np.abs(Atilde[k,:,:])
            Atilde_kii = np.sqrt(Atilde_k.diagonal()[:,np.newaxis])
            angles = (Atilde_k/Atilde_kii)/(Atilde_kii.T)
            
            prctiles = np.arange(100)
            ax[4]
            ax[4].cla()
            ax[4].plot(prctiles, np.percentile(angles.flatten(), prctiles), 'bo-')
            ax[4].plot([0,100], [0,0], 'g-')
            ax[4].plot([0,100], [angles[Psitilde_i[m,0],Psitilde_i[m,1]]]*2, 'r-')
            ax[4].set_xlabel('percentiles')
            ax[4].set_title('$|\\widetilde{A}_{%dij}|/(\\widetilde{A}_{%dii}\\widetilde{A}_{%djj})$' % (k, k, k))
            
            
            local_scales = gamma[[k],:].T*Atilde_kii
            #dlocal_scales = squareform(pdist(local_scales))
            dlocal_scales = np.log(local_scales/local_scales.T+1)
            
            ax[5]
            ax[5].cla()
            ax[5].plot(prctiles, np.percentile(dlocal_scales.flatten(), prctiles), 'bo-')
            ax[5].plot([0,100], [np.log(2)]*2, 'g-')
            ax[5].plot([0,100], [dlocal_scales[Psitilde_i[m,0],Psitilde_i[m,1]]]*2, 'r-')
            ax[5].set_xlabel('percentiles')
            ax[5].set_title('$\\log(\\gamma_{%di}\\sqrt{\\widetilde{A}_{%dii}} / \
                      \\gamma_{%dj}\\sqrt{\\widetilde{A}_{%djj}}+1)$' % (k, k, k, k))
                
            fig.canvas.draw()
            fig.canvas.flush_events()
            if self.save_dir:
                if not os.path.isdir(self.save_dir+'/intermediate_views'):
                    os.makedirs(self.save_dir+'/intermediate_views')
                plt.savefig(self.save_dir+'/intermediate_views/'+str(m)+'.png') 
        
    def compare_local_high_low_distortion(self, X, Atilde, Psi_gamma, Psi_i, zeta, save_subdir='', figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        is_3d_data = X.shape[1] == 3
        
        n = Atilde.shape[0]
        prctiles = np.arange(100)
        
        Atilde_ki_1i_2 = np.abs(Atilde[np.arange(n),Psi_i[:,0],Psi_i[:,1]])
        Atilde_ki_1i_1 = np.sqrt(Atilde[np.arange(n),Psi_i[:,0],Psi_i[:,0]])
        Atilde_ki_2i_2 = np.sqrt(Atilde[np.arange(n),Psi_i[:,1],Psi_i[:,1]])
        angles = (Atilde_ki_1i_2/Atilde_ki_1i_1)/(Atilde_ki_2i_2.T)
        
        local_scales_i_1 = Psi_gamma[np.arange(n),0].T*Atilde_ki_1i_1
        local_scales_i_2 = Psi_gamma[np.arange(n),1].T*Atilde_ki_2i_2
        dlocal_scales = np.log(local_scales_i_1/local_scales_i_2+1)
        
        fig = plt.figure(1, figsize=figsize)
        fig.canvas.mpl_connect('close_event', on_close)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        
        ax = []
        if is_3d_data:
            ax.append(fig.add_subplot(321, projection='3d'))
            ax.append(fig.add_subplot(322))
            ax.append(fig.add_subplot(323, projection='3d'))
            for i in range(3,6):
                ax.append(fig.add_subplot(321+i))
            p = ax[0].scatter(X[:,0], X[:,1], X[:,2], s=s, c=zeta, cmap='jet')
            ax[0].autoscale()
        else:
            for i in range(6):
                ax.append(fig.add_subplot(321+i))
            p = ax[0].scatter(X[:,0], X[:,1], s=s, c=zeta, cmap='jet')
            ax[0].axis('image')
        
        cb = plt.colorbar(p, ax=ax[0])
        ax[0].set_title('$x_k$ colored by $\\zeta_{kk}$')
        ax[3].axis('off')
            
        ax[1]
        ax[1].cla()
        ax[1].plot(prctiles, np.percentile(zeta, prctiles), 'bo-')
        ax[1].set_xlabel('percentiles')
        ax[1].set_title('$\\zeta_{kk}$\nDouble click = Select threshold\nPress button = quit')
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        while True:
            ax[1]
            
            to_exit = plt.waitforbuttonpress(timeout=20)
            if to_exit is None:
                print('Timed out')
                break
                
            if to_exit:
                plt.close()
                return
            
            zeta_k = plt.ginput(1)
            thresh = zeta_k[0][1]
            
            ax[1].cla()
            ax[1].plot(prctiles, np.percentile(zeta, prctiles), 'bo-')
            ax[1].plot([0,100], [thresh]*2, 'r-')
            ax[1].set_xlabel('percentiles')
            ax[1].set_title('$\\zeta_{kk}$, threshold = %f\nDouble click = Select threshold\nPress button = quit' % thresh)
            fig.canvas.draw()
            fig.canvas.flush_events()

            low_dist_mask = zeta <= thresh
            high_dist_mask = zeta >= thresh

            ax[2]
            ax[2].cla()
            if is_3d_data:
                ax[2].scatter(X[low_dist_mask,0], X[low_dist_mask,1], X[low_dist_mask,2], s=s, c='b')
                ax[2].scatter(X[high_dist_mask,0], X[high_dist_mask,1], X[high_dist_mask,2], s=s, c='r')
                ax[2].autoscale()
            else:
                ax[2].scatter(X[low_dist_mask,0], X[low_dist_mask,1], s=s, c='b')
                ax[2].scatter(X[high_dist_mask,0], X[high_dist_mask,1], s=s, c='r')
                ax[2].axis('image')
            
            ax[2].set_title('blue = low distortion, Red = high distortion')

            ax[4]
            ax[4].cla() 
            ax[4].boxplot([angles[low_dist_mask],angles[high_dist_mask]],
                        labels=['low $\\zeta_{kk}$','high $\\zeta_{kk}$'], notch=True,
                               vert=False, patch_artist=True)
            ax[4].set_title('$|\\widetilde{A}_{ki_1i_2}|/(\\widetilde{A}_{ki_1i_1}\\widetilde{A}_{ki_2i_2})$')

            ax[5]
            ax[5].cla()
            ax[5].boxplot([dlocal_scales[low_dist_mask],dlocal_scales[high_dist_mask]],
                        labels=['low $\\zeta_{kk}$','high $\\zeta_{kk}$'], notch=True,
                               vert=False, patch_artist=True)
            ax[5].set_title('$\\log(\\gamma_{ki_1}\\sqrt{\\widetilde{A}_{ki_1i_1}} / \
                      \\gamma_{ki_2}\\sqrt{\\widetilde{A}_{ki_2i_2}}+1)$')
            fig.canvas.draw()
            fig.canvas.flush_events()
            if self.save_dir:
                if not os.path.isdir(self.save_dir+'/local_high_low_distortion/'+save_subdir):
                    os.makedirs(self.save_dir+'/local_high_low_distortion/'+save_subdir)
                plt.savefig(self.save_dir+'/local_high_low_distortion/'+save_subdir+'/thresh='+str(thresh)+'.png') 
            
        
    def compare_intermediate_high_low_distortion(self, X, Atilde, Psitilde_gamma, 
                                                 Psitilde_i, zetatilde, c, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        is_3d_data = X.shape[1] == 3

        zeta = zetatilde[c]

        n = Atilde.shape[0]
        M = Psitilde_i.shape[0]
        prctiles = np.arange(100)

        Atilde_ki_1i_2 = np.abs(Atilde[np.arange(M),Psitilde_i[:,0],Psitilde_i[:,1]])
        Atilde_ki_1i_1 = np.sqrt(Atilde[np.arange(M),Psitilde_i[:,0],Psitilde_i[:,0]])
        Atilde_ki_2i_2 = np.sqrt(Atilde[np.arange(M),Psitilde_i[:,1],Psitilde_i[:,1]])
        angles = (Atilde_ki_1i_2/Atilde_ki_1i_1)/(Atilde_ki_2i_2.T)

        local_scales_i_1 = Psitilde_gamma[np.arange(M),0].T*Atilde_ki_1i_1
        local_scales_i_2 = Psitilde_gamma[np.arange(M),1].T*Atilde_ki_2i_2
        dlocal_scales = np.log(local_scales_i_1/local_scales_i_2+1)-np.log(2)

        fig = plt.figure(1, figsize=figsize)
        fig.canvas.mpl_connect('close_event', on_close)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        ax = []
        if is_3d_data:
            ax.append(fig.add_subplot(321, projection='3d'))
            ax.append(fig.add_subplot(322))
            ax.append(fig.add_subplot(323, projection='3d'))
            for i in range(3,6):
                ax.append(fig.add_subplot(321+i))
                
            p = ax[0].scatter(X[:,0], X[:,1], X[:,2], s=s, c=zeta, cmap='jet')
            ax[0].autoscale()
        else:
            for i in range(6):
                ax.append(fig.add_subplot(321+i))
            p = ax[0].scatter(X[:,0], X[:,1], s=s, c=zeta, cmap='jet')
            ax[0].axis('image')
            
        cb = plt.colorbar(p, ax=ax[0])
        ax[0].set_title('$x_k$ colored by $\\widetilde{\\zeta}_{c_kc_k}$')
        ax[3].axis('off')

        ax[1]
        ax[1].cla()
        ax[1].plot(prctiles, np.percentile(zetatilde, prctiles), 'bo-')
        ax[1].set_xlabel('percentiles')
        ax[1].set_title('$\\widetilde{\\zeta}_{mm}$\nDouble click = Select threshold\nPress button = quit')
        fig.canvas.draw()
        fig.canvas.flush_events()

        while True:
            ax[1]
            to_exit = plt.waitforbuttonpress(timeout=20)
            if to_exit is None:
                print('Timed out')
                break

            if to_exit:
                plt.close()
                return

            zetatilde_m = plt.ginput(1)
            thresh = zetatilde_m[0][1]

            ax[1].cla()
            ax[1].plot(prctiles, np.percentile(zetatilde, prctiles), 'bo-')
            ax[1].plot([0,100], [thresh]*2, 'r-')
            ax[1].set_xlabel('percentiles')
            ax[1].set_title('$\\widetilde{\\zeta}_{mm}$, threshold = %f\nDouble click = Select threshold\nPress button = quit' % thresh)
            fig.canvas.draw()
            fig.canvas.flush_events()

            low_dist_mask = zeta <= thresh
            high_dist_mask = zeta >= thresh

            ax[2]
            ax[2].cla()
            if is_3d_data:
                ax[2].scatter(X[low_dist_mask,0], X[low_dist_mask,1], X[low_dist_mask,2], s=s, c='b')
                ax[2].scatter(X[high_dist_mask,0], X[high_dist_mask,1], X[high_dist_mask,2], s=s, c='r')
                ax[2].autoscale()
            else:
                ax[2].scatter(X[low_dist_mask,0], X[low_dist_mask,1], s=s, c='b')
                ax[2].scatter(X[high_dist_mask,0], X[high_dist_mask,1], s=s, c='r')
                ax[2].axis('image')
            
            ax[2].set_title('blue = low distortion, Red = high distortion')

            low_dist_mask = zetatilde <= thresh
            high_dist_mask = zetatilde >= thresh

            ax[4]
            ax[4].cla() 
            ax[4].boxplot([angles[low_dist_mask],angles[high_dist_mask]],
                            labels=['low $\\widetilde{\\zeta}_{mm}$','high $\\widetilde{\\zeta}_{mm}$'], notch=True,
                                   vert=False, patch_artist=True)
            ax[4].set_title('$|\\widetilde{A}_{ki_1i_2}|/(\\widetilde{A}_{ki_1i_1}\\widetilde{A}_{ki_2i_2})$')

            ax[5]
            ax[5].cla()
            ax[5].boxplot([dlocal_scales[low_dist_mask],dlocal_scales[high_dist_mask]],
                            labels=['low $\\widetilde{\\zeta}_{mm}$','high $\\widetilde{\\zeta}_{mm}$'], notch=True,
                               vert=False, patch_artist=True)
            ax[5].set_title('$\\log(\\gamma_{ki_1}\\sqrt{\\widetilde{A}_{ki_1i_1}} / \
                      \\gamma_{ki_2}\\sqrt{\\widetilde{A}_{ki_2i_2}}+1)$')
            fig.canvas.draw()
            fig.canvas.flush_events()
            if self.save_dir:
                if not os.path.isdir(self.save_dir+'/intermediate_high_low_distortion'):
                    os.makedirs(self.save_dir+'/intermediate_high_low_distortion')
                plt.savefig(self.save_dir+'/intermediate_high_low_distortion/thresh='+str(thresh)+'.png') 
    
    def seq_of_intermediate_views(self, X, c, seq, rho, Utilde, figsize=None, s=20):
        seq = np.copy(seq)
        M = seq.shape[0]
        mu = np.zeros((M,X.shape[1]))
        source = np.zeros((M-1,X.shape[1]))
        comp = np.zeros((M-1,X.shape[1]))
        for m in range(M):
            mu[m,:] = np.mean(X[Utilde[m,:],:],0)
        
        for m in range(1,M):
            source[m-1,:] = mu[rho[seq[m]],:]
            comp[m-1,:] = mu[seq[m],:]-source[m-1,:]
         
        seq[seq] = np.arange(M)
        seq = seq[c]
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], s=s, c=seq, cmap='jet')
            plt.axis('image')
            plt.colorbar()
            plt.quiver(source[:,0], source[:,1], comp[:,0], comp[:,1], np.arange(M-1),
                       cmap='gray', angles='xy', scale_units='xy', scale=1)
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            ax.autoscale()
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=seq, cmap='jet')
            fig.colorbar(p)
        plt.title('Views colored in the sequence they are visited')
        if self.save_dir:
            plt.savefig(self.save_dir+'/seq_in_which_views_are_visited.png') 
    
    def global_embedding(self, y, labels, cmap0, color_of_pts_on_tear=None, cmap1=None,
                         title=None, figsize=None, s=30):
        d = y.shape[1]
        if d > 3:
            return
        
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        
        if d == 2:
            ax = fig.add_subplot()
            ax.scatter(y[:,0], y[:,1], s=s, c=labels, cmap=cmap0)
            ax.axis('image')
        elif d ==3:
            ax = fig.add_subplot(projection='3d')
            ax.scatter(y[:,0], y[:,1], y[:,2], s=s, c=labels, cmap=cmap0)
        
        if color_of_pts_on_tear is not None:
            if d == 2:
                ax.scatter(y[:,0], y[:,1],
                           s=s, c=color_of_pts_on_tear, cmap=cmap1)
            elif d==3:
                ax.scatter(y[:,0], y[:,1], y[:,2],
                           s=s, c=color_of_pts_on_tear, cmap=cmap1)
        ax.axis('off')
        if title is not None:
            ax.set_title(title)
            
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/ge'):
                os.makedirs(self.save_dir+'/ge')
            plt.savefig(self.save_dir+'/ge/'+str(title)+'.png')
    
    def global_embedding_images(self, X, img_shape, y, labels, cmap0, color_of_pts_on_tear=None, cmap1=None,
                         title=None, figsize=None, s=30, zoom=1):
        d = y.shape[1]
        if d > 2:
            return
        
        fig = plt.figure(figsize=figsize)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        
        ax = fig.add_subplot()
        ax.scatter(y[:,0], y[:,1], s=s, c=labels, cmap=cmap0)
        ax.axis('image')
        if color_of_pts_on_tear is not None:
            ax.scatter(y[:,0], y[:,1],
                       s=s, c=color_of_pts_on_tear, cmap=cmap1)

        ax.set_title(title)
        ax.axis('off')
        while True:
            y_k = plt.ginput(1)
            if len(y_k)==0:
                break
            y_k = np.array(y_k[0])[np.newaxis,:]
            k = np.argmin(np.sum((y-y_k)**2,1))

            #ax.plot(y[k,0], y[k,1], 'ro', markersize=10)
            imscatter(y[k,0], y[k,1], X[k,:].reshape(img_shape).T, ax=ax, zoom=zoom)

            if self.save_dir:
                if not os.path.isdir(self.save_dir+'/ge_img'):
                    os.makedirs(self.save_dir+'/ge_img')
                plt.savefig(self.save_dir+'/ge_img/'+str(title)+'.png')