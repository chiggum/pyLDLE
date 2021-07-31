import sys
import os

import numpy as np

import matplotlib
print('matplotlib.get_backend() = ', matplotlib.get_backend())
#matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
plt.rcParams.update({'scatter.marker':'.'})
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from scipy.spatial.distance import pdist, squareform

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

# colorcube colormap taken from matlab
def colorcube(m):
    nrgsteps = np.fix(np.power(m,1/3)+np.finfo(np.float32).eps)
    extra = m-np.power(nrgsteps,3)
    if (extra == 0) and (nrgsteps > 2):
        nbsteps = nrgsteps - 1
    else:
        nbsteps = nrgsteps

    rgstep = 1/(nrgsteps-1)
    bstep  = 1/(nbsteps-1)
    [r,g,b] = np.meshgrid(np.arange(nrgsteps)*rgstep,
                          np.arange(nrgsteps)*rgstep,
                          np.arange(nbsteps)*bstep)
    r = r.flatten('F')[:,np.newaxis]
    g = g.flatten('F')[:,np.newaxis]
    b = b.flatten('F')[:,np.newaxis]
    mymap = np.concatenate([r,g,b], axis=1)
    
    

    diffmap = np.diff(mymap.T, axis=0).T
    summap = np.sum(np.abs(diffmap),1)
    notgrays = (summap != 0)
    mymap = mymap[notgrays,:]

    summap = np.concatenate([np.sum(mymap[:,[0,1]],1)[:,np.newaxis],
                             np.sum(mymap[:,[1,2]],1)[:,np.newaxis],
                             np.sum(mymap[:,[0,2]],1)[:,np.newaxis]], axis=1)
    mymap = mymap[np.min(summap,axis=1) != 0,:]
    
    remlen = m - mymap.shape[0] - 1

    rgbnsteps = np.floor(remlen / 4)
    knsteps   = remlen - 3*rgbnsteps

    rgbstep = 1/(rgbnsteps)
    kstep   = 1/(knsteps  )

    rgbramp = np.arange(0,rgbnsteps)*rgbstep + rgbstep
    rgbzero = np.zeros((rgbramp.shape[0], 1))
    kramp   = np.arange(0,knsteps)*kstep + kstep
    
    rgbramp = rgbramp[:,np.newaxis]
    kramp = kramp[:,np.newaxis]

    mymap = np.concatenate([mymap,
                            np.concatenate([rgbramp, rgbzero, rgbzero], axis=1),
                            np.concatenate([rgbzero, rgbramp, rgbzero], axis=1),
                            np.concatenate([rgbzero, rgbzero, rgbramp], axis=1),
                            np.zeros((1,3)),
                            np.concatenate([kramp, kramp, kramp,], axis=1)], axis=0)
    return mymap

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

def closest_pt(x_min, x_max, y_min, y_max, p):
    i = np.argmin([p[0]-x_min,x_max-p[0], p[1]-y_min, y_max-p[1]])
    if i == 0:
        return [x_min,p[1]]
    elif i==1:
        return [x_max,p[1]]
    elif i==2:
        return [p[0],y_min]
    elif i==3:
        return [p[0],y_max]

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

class Visualize:
    def __init__(self, save_dir=''):
        self.save_dir = save_dir
        if self.save_dir:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
        pass
    
    def data(self, X, labels, title='Data', figsize=None, s=20, cmap='jet'):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], s=s, c=labels, cmap=cmap)
            plt.axis('image')
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=labels, cmap=cmap)
            set_axes_equal(ax)
        plt.title(title)
        plt.tight_layout()
        plt.axis('off')
        if self.save_dir:
            plt.savefig(self.save_dir+'/' + title + '.png') 
        
    def eigenvalues(self, lmbda, figsize=None):
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
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
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], s=s, c=gamma[:,i], cmap='jet')
            plt.axis('image')
            plt.colorbar()
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=gamma[:,i], cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p)
        plt.title('$\gamma_{%d}$'%i)
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/gamma'):
                os.makedirs(self.save_dir+'/gamma')
            plt.savefig(self.save_dir+'/gamma/'+str(i)+'.png') 
    
    def eigenvector(self, X, phi, i, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], s=s, c=phi[:,i], cmap='jet')
            plt.axis('image')
            plt.colorbar()
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=phi[:,i], cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p)
        plt.title('$\phi_{%d}$'%i)
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/eigvecs'):
                os.makedirs(self.save_dir+'/eigvecs')
            plt.savefig(self.save_dir+'/eigvecs/'+str(i)+'.png') 
    
    def grad_phi(self, X, phi, grad_phi, i, prop=0.01, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
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
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=phi[:,i], cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p, ax=ax)
            plt.title('$\phi_{%d}$'%i)
            ax = fig.add_subplot(122,projection='3d')
            p = ax.quiver(X[mask,0], X[mask,1], X[mask,2], grad_phi[mask,i,0], grad_phi[mask,i,1], grad_phi[mask,i,2])
            set_axes_equal(ax)
            plt.title('$\\nabla\phi_{%d}$'%i)
        
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/grad_phi'):
                os.makedirs(self.save_dir+'/grad_phi')
            plt.savefig(self.save_dir+'/grad_phi/'+str(i)+'.png') 
    
    def Atilde(self, X, phi, i, j, Atilde, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
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
            set_axes_equal(ax)
            fig.colorbar(p, ax=ax)
            plt.title('$\phi_{%d}$'%i)
            ax = fig.add_subplot(132,projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=phi[:,i], cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p, ax=ax)
            plt.title('$\phi_{%d}$'%j)
            ax = fig.add_subplot(133,projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=Atilde[:,i,j], cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p, ax=ax)
            plt.title('$\widetilde{A}_{:%d%d}$'%(i,j))
        
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/Atilde'):
                os.makedirs(self.save_dir+'/Atilde')
            plt.savefig(self.save_dir+'/Atilde/'+str(i)+'_'+str(j)+'.png') 
    
    def n_eigvecs_w_grad_lt(self, X, Atilde, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        
        if X.shape[1] == 3:
            ax = fig.add_subplot(122,projection='3d')
        elif X.shape[1] == 2:
            ax = fig.add_subplot(122)
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
                ax.cla()
                p = ax.scatter(X[:,0], X[:,1], s=s, c=n_grad_lt, cmap='jet')
                ax.axis('image')
                if cb is not None:
                    cb.remove()
                cb = fig.colorbar(p, ax=ax)
                ax.set_title('$n_k = \sum_{i}\widetilde{A}_{kii} < %f$'% thresh)
            elif X.shape[1] == 3:
                ax.cla()
                ax.autoscale()
                p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=n_grad_lt, cmap='jet')
                set_axes_equal(ax)
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
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        if X.shape[1] == 1:
            plt.scatter(X[:,0], X[:,0], s=s, c=zeta, cmap='jet')
            plt.axis('image')
            plt.colorbar()
        elif X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], s=s, c=zeta, cmap='jet')
            plt.axis('image')
            plt.colorbar()
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=zeta, cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p)
        plt.title(title)
        if self.save_dir:
            plt.savefig(self.save_dir+'/'+title+'.png') 
    
    def distortion_boxplot(self, zeta, title, figsize=None):
        fig = plt.figure(figsize=figsize)
        plt.boxplot([zeta],labels=[title], notch=True, patch_artist=True)
        if self.save_dir:
            plt.savefig(self.save_dir+'/box_'+title+'.png') 
    
    def dX(self, X, ddX, title, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
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
            set_axes_equal(ax)
            fig.colorbar(p, ax=ax)
            ax.set_title('distance from dX')
            ax = fig.add_subplot(122, projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=ddX==0, cmap='jet')
            set_axes_equal(ax)
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
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=chi, cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p)
        plt.title('\chi')
    
    def chosen_eigevec_inds_for_local_views(self, X, Psi_i, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(1, figsize=figsize)
        fig.tight_layout()
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.subplot(211)
            plt.scatter(X[:,0], X[:,1], s=s, c=Psi_i[:,0], cmap='jet')
            plt.axis('image')
            plt.colorbar()
            plt.title('$\\phi_{i_1}$')
            plt.subplot(212)
            plt.scatter(X[:,0], X[:,1], s=s, c=Psi_i[:,1], cmap='jet')
            plt.axis('image')
            plt.colorbar()
            plt.title('$\\phi_{i_2}$')
        elif X.shape[1] == 3:
            ax = fig.add_subplot(211, projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=Psi_i[:,0], cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p)
            ax.set_title('$\\phi_{i_1}$')
            ax = fig.add_subplot(212, projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=Psi_i[:,1], cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p)
            ax.set_title('$\\phi_{i_2}$')
        
        if self.save_dir:
            plt.savefig(self.save_dir+'/chosen_eigvecs_for_local_views.png') 
    
    def chosen_eigevec_inds_for_intermediate_views(self, X, Psitilde_i, c, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(1, figsize=figsize)
        fig.tight_layout()
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.subplot(211)
            plt.scatter(X[:,0], X[:,1], s=s, c=Psitilde_i[c,0], cmap='jet')
            plt.axis('image')
            plt.colorbar()
            plt.title('$\\phi_{i_1}$')
            plt.subplot(212)
            plt.scatter(X[:,0], X[:,1], s=s, c=Psitilde_i[c,1], cmap='jet')
            plt.axis('image')
            plt.colorbar()
            plt.title('$\\phi_{i_2}$')
        elif X.shape[1] == 3:
            ax = fig.add_subplot(211, projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=Psitilde_i[c,0], cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p)
            plt.title('$\\phi_{i_1}$')
            ax = fig.add_subplot(212, projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=Psitilde_i[c,1], cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p)
            ax.set_title('$\\phi_{i_2}$')
        
        if self.save_dir:
            plt.savefig(self.save_dir+'/chosen_eigvecs_for_intermediate_views.png') 
    
    def local_views(self, X, phi, U, gamma, Atilde, Psi_gamma, Psi_i, zeta, k=None, save_subdir='', figsize=(8,15), s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        is_3d_data = X.shape[1] == 3
        n,N = phi.shape
        
        fig = plt.figure(1, figsize=figsize)
        fig.tight_layout()
        fig.canvas.mpl_connect('close_event', on_close)
        if matplotlib.get_backend().startswith('Qt'):
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
            set_axes_equal(ax[0])
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
            fig = plt.figure(1, figsize=figsize)
            fig.tight_layout()
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
                set_axes_equal(ax[0])
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
            ax[3].set_title('$\\zeta_{%d%d}=%.3f$\n'\
                              '$\\Phi_{%d}(\\mathcal{M})$ in red\n$\\Phi_{%d}(U_{%d})$ in black'\
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
                    set_axes_equal(ax[j+1])
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
            ax[4].set_title('$\\cos(\\nabla\\phi_{i_1},\\nabla\\phi_{i_2})$')
            
            
            local_scales = gamma[[k],:].T*Atilde_kii
            #dlocal_scales = squareform(pdist(local_scales))
            dlocal_scales = np.log(local_scales/local_scales.T+1)
            
            ax[5]
            ax[5].cla()
            ax[5].plot(prctiles, np.percentile(dlocal_scales.flatten(), prctiles), 'bo-')
            ax[5].plot([0,100], [np.log(2)]*2, 'g-')
            ax[5].plot([0,100], [dlocal_scales[Psi_i[k,0],Psi_i[k,1]]]*2, 'r-')
            ax[5].set_xlabel('percentiles')
            ax[5].set_title('$\\log(\\gamma_{i_1}\\left\\|\\nabla\\phi_{i_1}\\right\\|_2 / \
                      \\gamma_{i_2}\\left\\|\\nabla\\phi_{i_2}\\right\\|_2+1)$')
                
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
        if matplotlib.get_backend().startswith('Qt'):
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
            set_axes_equal(ax[0])
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
                set_axes_equal(ax[0])
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
                    set_axes_equal(ax[j+1])
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
        if matplotlib.get_backend().startswith('Qt'):
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
            set_axes_equal(ax[0])
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
                set_axes_equal(ax[0])
            else:
                p = ax[0].scatter(X[:,0], X[:,1], s=s*(1-Utilde_m), c=zeta, cmap='jet')
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
                    set_axes_equal(ax[j+1])
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
        if matplotlib.get_backend().startswith('Qt'):
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
            set_axes_equal(ax[0])
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
                set_axes_equal(ax[2])
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
                if not os.path.isdir(self.save_dir+'/local_high_low_distortion/'):
                    os.makedirs(self.save_dir+'/local_high_low_distortion/')
                plt.savefig(self.save_dir+'/local_high_low_distortion/thresh='+str(thresh)+'.png') 
            
        
    def compare_intermediate_high_low_distortion(self, X, Atilde, Psitilde_gamma, 
                                                 Psitilde_i, zetatilde, c,  save_subdir='', figsize=None, s=20):
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
        if matplotlib.get_backend().startswith('Qt'):
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
            set_axes_equal(ax[0])
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
                set_axes_equal(ax[2])
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
    
    def seq_of_intermediate_views(self, X, c, seq, rho, Utilde, figsize=None, s=20, cmap='jet'):
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
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        if X.shape[1] == 2:
            if cmap == 'colorcube':
                c_ = colorcube(M)[seq,:]
                plt.scatter(X[:,0], X[:,1], s=s, c=c_)
            else:
                plt.scatter(X[:,0], X[:,1], s=s, c=seq, cmap=cmap)
            plt.axis('image')
            plt.colorbar()
            plt.quiver(source[:,0], source[:,1], comp[:,0], comp[:,1], np.arange(M-1),
                       cmap='gray', angles='xy', scale_units='xy', scale=1)
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            if cmap == 'colorcube':
                c_ = colorcube(M)[seq,:]
                p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=c_)
            else:
                p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=seq, cmap=cmap)
            set_axes_equal(ax)
            fig.colorbar(p)
        plt.title('Views colored in the sequence they are visited')
        if self.save_dir:
            plt.savefig(self.save_dir+'/seq_in_which_views_are_visited.png') 
    
    def global_embedding(self, y, labels, cmap0, color_of_pts_on_tear=None, cmap1=None,
                         title=None, figsize=None, s=30, set_title=False):
        d = y.shape[1]
        if d == 1:
            y = np.concatenate([y,y],axis=1)
        d = y.shape[1]
        if d > 3:
            return
        
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        
        if d == 2:
            ax = fig.add_subplot()
            ax.scatter(y[:,0], y[:,1], s=s, c=labels, cmap=cmap0)
            ax.axis('image')
        elif d == 3:
            ax = fig.add_subplot(projection='3d')
            ax.scatter(y[:,0], y[:,1], y[:,2], s=s, c=labels, cmap=cmap0)
            set_axes_equal(ax)
        
        if (color_of_pts_on_tear is not None) and (np.sum(np.isnan(color_of_pts_on_tear)) < y.shape[0]):
            if cmap1 == 'colorcube':
                uniq_vals = np.sort(np.unique(color_of_pts_on_tear[~np.isnan(color_of_pts_on_tear)]))
                mymap = {}
                ctr = 0
                for i in range(uniq_vals.shape[0]):
                    mymap[uniq_vals[i]] = i

                cc_map = colorcube(uniq_vals.shape[0])
                n_ = color_of_pts_on_tear.shape[0]
                c_ = np.ones((n_,4))
                for i in range(n_):
                    if ~np.isnan(color_of_pts_on_tear[i]):
                        c_[i,:3] = cc_map[mymap[color_of_pts_on_tear[i]],:]
                    else:
                        c_[i,3] = 0

                if d == 2:
                    ax.scatter(y[:,0], y[:,1], s=s, c=c_)
                elif d==3:
                    ax.scatter(y[:,0], y[:,1], y[:,2], s=s, c=c_)
                    set_axes_equal(ax)
            else:
                if d == 2:
                    ax.scatter(y[:,0], y[:,1],
                               s=s, c=color_of_pts_on_tear, cmap=cmap1)
                elif d==3:
                    ax.scatter(y[:,0], y[:,1], y[:,2],
                               s=s, c=color_of_pts_on_tear, cmap=cmap1)
                    set_axes_equal(ax)
        ax.axis('off')
        if set_title:
            ax.set_title(title)
        
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/ge'):
                os.makedirs(self.save_dir+'/ge')
            plt.savefig(self.save_dir+'/ge/'+str(title)+'.png', bbox_inches = 'tight',pad_inches = 0)
    
    def global_embedding_images(self, X, img_shape, y, labels, cmap0, color_of_pts_on_tear=None, cmap1=None,
                         title=None, figsize=None, s=30, zoom=1, offset_ratio=0.2,w_ratio=0.0025):
        d = y.shape[1]
        if d > 2:
            return
        
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        
        ax = fig.add_subplot()
        
        ax = fig.add_subplot()
        y_x_max = np.max(y[:,0])
        y_x_min = np.min(y[:,0])
        y_y_max = np.max(y[:,1])
        y_y_min = np.min(y[:,1])
        x_width = np.abs(y_x_max - y_x_min)
        y_width = np.abs(y_y_max - y_y_min)
        
        y_x_max = y_x_max + offset_ratio*x_width
        y_x_min = y_x_min - offset_ratio*x_width
        y_y_max = y_y_max + offset_ratio*y_width
        y_y_min = y_y_min - offset_ratio*y_width
        
        ax.plot([y_x_min,y_x_max],[y_y_min,y_y_min], 'k-', linewidth=0.5)
        ax.plot([y_x_min,y_x_max],[y_y_max,y_y_max], 'k-', linewidth=0.5)
        
        ax.plot([y_x_min,y_x_min],[y_y_min,y_y_max], 'k-', linewidth=0.5)
        ax.plot([y_x_max,y_x_max],[y_y_min,y_y_max], 'k-', linewidth=0.5)
        
        
        ax.scatter(y[:,0], y[:,1], s=s, c=labels, cmap=cmap0)
        ax.axis('image')
        if color_of_pts_on_tear is not None:
            ax.scatter(y[:,0], y[:,1],
                       s=s, c=color_of_pts_on_tear, cmap=cmap1)

        ax.set_title(title)
        #ax.axis('off')
        ax.set_title('Duble click to choose a point. Press button to exit')
        fig.canvas.draw()
        fig.canvas.flush_events()
        first_time = 1
        while True:
            to_exit = plt.waitforbuttonpress(timeout=20)
            if to_exit is None:
                print('Timed out')
                break

            if to_exit:
                plt.close()
                return
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            y_k = plt.ginput(1)
            if len(y_k)==0:
                break
            y_k = np.array(y_k[0])[np.newaxis,:]
            k = np.argmin(np.sum((y-y_k)**2,1))

            #ax.plot(y[k,0], y[k,1], 'ro', markersize=10)
            ax.set_title('Choose location of image.')
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            
            z = plt.ginput(1)
            if len(z)==0:
                break
            z = np.array(z[0])[np.newaxis,:]
            
            z = closest_pt(y_x_min, y_x_max, y_y_min, y_y_max, z.flatten().tolist())
            
            #if first_time:
            #    first_time = 0
            #    continue
                
            imscatter(z[0], z[1], X[k,:].reshape(img_shape).T, ax=ax, zoom=zoom)
            
            #ax.set_title('Choose arrow end.')
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            #z = plt.ginput(1)
            #if len(z)==0:
            #    break
            #z = np.array(z[0])[np.newaxis,:]
            
            ax.quiver(y[k,0],y[k,1],z[0]-y[k,0],z[1]-y[k,1],units='xy',scale=1,width=w_ratio*x_width)
            
            ax.set_title('Duble click to choose a point. Press button to exit')
            fig.canvas.draw()
            fig.canvas.flush_events()
            if self.save_dir:
                if not os.path.isdir(self.save_dir+'/ge_img'):
                    os.makedirs(self.save_dir+'/ge_img')
                plt.savefig(self.save_dir+'/ge_img/'+str(title)+'.png')
    
    def global_embedding_images_v2(self, X, img_shape, y, labels, cmap0, color_of_pts_on_tear=None, cmap1=None,
                         title='images', offset_ratio=0.3, zoom=1, nx=8, ny=8, v_ratio=0.8, w_ratio=0.005,
                         figsize=None, s=30, to_remove=False, k_to_avoid=[], to_T=True):
        d = y.shape[1]
        if d > 2:
            return
        
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        
        ax = fig.add_subplot()
        y_x_max = np.max(y[:,0])
        y_x_min = np.min(y[:,0])
        y_y_max = np.max(y[:,1])
        y_y_min = np.min(y[:,1])
        x_width = np.abs(y_x_max - y_x_min)
        y_width = np.abs(y_y_max - y_y_min)
        
        y_x_max = y_x_max + offset_ratio*x_width
        y_x_min = y_x_min - offset_ratio*x_width
        y_y_max = y_y_max + offset_ratio*y_width
        y_y_min = y_y_min - offset_ratio*y_width
        
        
        x_i = np.linspace(y_x_min, y_x_max, nx)
        y_i = np.linspace(y_y_min, y_y_max, ny)
        
        pts = np.concatenate([np.concatenate([x_i[:,np.newaxis], np.zeros((nx,1))+y_y_min], axis=1),
                             np.concatenate([x_i[:,np.newaxis], np.zeros((nx,1))+y_y_max], axis=1),
                             np.concatenate([np.zeros((ny,1))+y_x_min, y_i[:,np.newaxis]], axis=1),
                             np.concatenate([np.zeros((ny,1))+y_x_max, y_i[:,np.newaxis]], axis=1)],
                            axis=0)
        
        ax.scatter(y[:,0], y[:,1], s=s, c=labels, cmap=cmap0)
        ax.axis('image')
        if color_of_pts_on_tear is not None:
            ax.scatter(y[:,0], y[:,1],
                       s=s, c=color_of_pts_on_tear, cmap=cmap1)

        #ax.set_title(title)
        
        ax.set_title('Duble click to choose center. Press button to exit')
        fig.canvas.draw()
        fig.canvas.flush_events()
        to_exit = plt.waitforbuttonpress(timeout=20)
        if to_exit is None:
            print('Timed out')
            return

        if to_exit:
            plt.close()
            return
        
        y_bar = plt.ginput(1)
        if len(y_bar)==0:
            return
        y_bar = np.array(y_bar[0])[np.newaxis,:]
        
        ax.set_title(title)
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        if color_of_pts_on_tear is not None:
            mask = ~np.isnan(color_of_pts_on_tear)
            inds = np.where(mask)[0]
            V2 = y[mask,:]-y_bar
            V2 = V2/np.sqrt(np.sum(V2**2,1)[:,np.newaxis])
        else:
            is_visited = np.ones((y.shape[0]))
        
        for k_ in range(pts.shape[0]):
            if color_of_pts_on_tear is not None:
                v1 = pts[k_,:][np.newaxis,:] - y_bar
                v1 = v1/np.sqrt(np.sum(v1**2))
                cos_sim = np.sum(V2 * v1,1)
                k = np.argmax(cos_sim)
                k = inds[k]
            else:
                k = np.argmin(is_visited * np.sum((y-pts[k_:k_+1,:])**2,1))
                is_visited[k] = np.inf
            
            if k in k_to_avoid:
                continue
            
            #ax.plot(pts[k_,0], pts[k_,1], 'r.')
            if to_T:
                imscatter(pts[k_,0], pts[k_,1], X[k,:].reshape(img_shape).T, ax=ax, zoom=zoom)
            else:
                imscatter(pts[k_,0], pts[k_,1], X[k,:].reshape(img_shape), ax=ax, zoom=zoom)
            ax.quiver(y[k,0],y[k,1],v_ratio*(pts[k_,0]-y[k,0]),v_ratio*(pts[k_,1]-y[k,1]),
                      units='xy',scale=1,width=w_ratio*x_width)
        
        ax.axis('off')
        
        if to_remove:
            ax.set_title('Double click to choose a point. Press button to exit')
            while True:
                to_exit = plt.waitforbuttonpress(timeout=20)
                if to_exit is None:
                    print('Timed out')
                    break

                if to_exit:
                    #plt.close()
                    break

                fig.canvas.draw()
                fig.canvas.flush_events()

                y_k = plt.ginput(1)
                if len(y_k)==0:
                    break
                y_k = np.array(y_k[0])[np.newaxis,:]
                k = np.argmin(np.sum((y-y_k)**2,1))
                k_to_avoid.append(k)
                ax.plot(y[k,0], y[k,1], 'ro')

            print(k_to_avoid)
            print('Re run by passing printed k_to_avoid as argument')
        
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/ge_img_v2'):
                os.makedirs(self.save_dir+'/ge_img_v2')
            plt.savefig(self.save_dir+'/ge_img_v2/'+str(title)+'.png')