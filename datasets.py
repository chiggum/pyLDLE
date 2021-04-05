import numpy as np

class Datasets:
    def __init__(self):
        pass
    
    def rectangleGrid(self, ar=16, RES=100):
        sideLx = np.sqrt(ar)
        sideLy = 1/sideLx
        RESx = sideLx*RES+1
        RESy = sideLy*RES+1
        x = np.linspace(0, sideLx, RESx)
        y = np.linspace(0, sideLy, RESy)
        xv, yv = np.meshgrid(x, y);
        xv = xv.flatten('F')[:,np.newaxis]
        yv = yv.flatten('F')[:,np.newaxis]
        X = np.concatenate([xv,yv], axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat
    
    def sphere(self, n=10000):
        R = np.sqrt(1/(4*np.pi))
        indices = np.arange(n)+0.5
        phiv = np.arccos(1 - 2*indices/n)
        phiv = phiv[:,np.newaxis]
        thetav = np.pi*(1 + np.sqrt(5))*indices
        thetav = thetav[:,np.newaxis]
        X = np.concatenate([np.sin(phiv)*np.cos(thetav),
                            np.sin(phiv)*np.sin(thetav),
                            np.cos(phiv)], axis=1)
        X = X*R;
        labelsMat = np.concatenate([thetav, phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat