import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve

class Datasets:
    def __init__(self):
        pass
    
    def rectanglegrid(self, ar=16, RES=100):
        sideLx = np.sqrt(ar)
        sideLy = 1/sideLx
        RESx = int(sideLx*RES+1)
        RESy = int(sideLy*RES+1)
        x = np.linspace(0, sideLx, RESx)
        y = np.linspace(0, sideLy, RESy)
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,np.newaxis]
        yv = yv.flatten('F')[:,np.newaxis]
        X = np.concatenate([xv,yv], axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat
    
    def barbell(self, RES=100):
        A1 = 0.425
        Rmax = np.sqrt(A1/np.pi)
        sideL1x = 1.5
        sideL1y = (1-2*A1)/sideL1x

        sideLx = sideL1x+4*Rmax
        sideLy = 2*Rmax

        RESx = int(np.ceil(sideLx*RES)+1)
        RESy = int(np.ceil(sideLy*RES)+1)
        x1 = np.linspace(0,sideLx,RESx)
        y1 = np.linspace(0,sideLy,RESy)
        x1v, y1v = np.meshgrid(x1,y1);
        x1v = x1v.flatten('F')[:,np.newaxis]
        y1v = y1v.flatten('F')[:,np.newaxis]
        x2v = np.copy(x1v)
        y2v = np.copy(y1v)
        
        mask1 = (((x1v-Rmax)**2+(y1v-Rmax)**2) < Rmax**2)|(((x1v-3*Rmax-sideL1x)**2+(y1v-Rmax)**2)<Rmax**2)
        mask2 = (x2v>=(2*Rmax))&(x2v<=(2*Rmax+sideL1x))&(y2v>(Rmax-sideL1y/2))&(y2v<(Rmax+sideL1y/2))
        x1v = x1v[mask1][:,np.newaxis]
        y1v = y1v[mask1][:,np.newaxis]
        x2v = x2v[mask2][:,np.newaxis]
        y2v = y2v[mask2][:,np.newaxis]
        xv = np.concatenate([x1v,x2v],axis=0)
        yv = np.concatenate([y1v,y2v],axis=0)
        X = np.concatenate([xv,yv],axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat
    
    def squarewithtwoholes(self, RES=100):
        sideLx = 1
        sideLy = 1
        RESx = sideLx*RES+1
        RESy = sideLy*RES+1
        x = np.linspace(0,sideLx,RESx);
        y = np.linspace(0,sideLy,RESy);
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,np.newaxis]
        yv = yv.flatten('F')[:,np.newaxis]
        X = np.concatenate([xv,yv], axis=1)
        hole1 = np.sqrt((X[:,0] - 0.5*np.sqrt(2))**2 + (X[:,1]-0.5*np.sqrt(2))**2) < 0.1*np.sqrt(2)
        hole2 = np.abs(X[:,0] - 0.2*np.sqrt(2)) + np.abs(X[:,1]-0.2*np.sqrt(2)) < 0.1*np.sqrt(2)
        X = X[~hole1 & ~hole2,:]
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat
    
    def spherewithhole(self, n=10000):
        Rmax = np.sqrt(1/(4*np.pi))
        indices = np.arange(n)
        indices = indices+0.5
        phiv = np.arccos(1 - 2*indices/n)[:,np.newaxis]
        thetav = (np.pi*(1 + np.sqrt(5))*indices)[:,np.newaxis]
        X = np.concatenate([np.sin(phiv)*np.cos(thetav), np.sin(phiv)*np.sin(thetav), np.cos(phiv)], axis=1)
        X = X*Rmax
        z0 = np.max(X[:,2])
        R_hole = Rmax/6
        hole = (X[:,0]**2+X[:,1]**2+(X[:,2]-z0)**2)<R_hole**2
        X = X[~hole,:]
        thetav = thetav[~hole][:,np.newaxis]
        phiv = phiv[~hole][:,np.newaxis]
        labelsMat = np.concatenate([np.mod(thetav,2*np.pi), phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat
    
    def swissrollwithhole(self, RES=100):
        theta0 = 3*np.pi/2
        nturns = 2
        rmax = 2*1e-2
        sideL1 = integrate.quad(lambda x: rmax*np.sqrt(1+x**2), theta0, theta0*(1+nturns))[0]
        sideL2 = 1/sideL1
        RESt = int(np.ceil(sideL1*RES+1))
        tdistv = np.linspace(0,sideL1,RESt)
        tv = []
        for tdist in tdistv.tolist():
            tt = fsolve(lambda x: (0.5*rmax*(x*np.sqrt(1+x**2)+np.arcsinh(x)))-\
                                   0.5*rmax*(theta0*np.sqrt(1+theta0**2)+np.arcsinh(theta0))-\
                                   tdist,theta0*(1+nturns/2))
            tv.append(tt)
        tv = np.array(tv)    
        RESh = int(np.ceil(sideL2*RES+1))
        heightv = np.linspace(0,sideL2,RESh)[:,np.newaxis]
        heightv = np.tile(heightv,[RESt,1])
        heightv = heightv.flatten('F')[:,np.newaxis]
        tv = np.repeat(tv,RESh)[:,np.newaxis]
        X=np.concatenate([rmax*tv*np.cos(tv), heightv, rmax*tv*np.sin(tv)], axis=1)

        y_mid = sideL2*0.5
        t_min = np.min(tv)
        t_max = np.max(tv)
        t_range = t_max-t_min
        t_mid = t_min + t_range/2
        x_mid = rmax*t_mid*np.cos(t_mid)
        z_mid = rmax*t_mid*np.sin(t_mid)
        hole = np.sqrt((X[:,0]-x_mid)**2+(X[:,1]-y_mid)**2+(X[:,2]-z_mid)**2)<0.1

        X = X[~hole,:]
        tv = tv[~hole]
        labelsMat = np.concatenate([tv, X[:,[1]]], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat
    
    def noisyswissroll(self, RES=100):
        theta0 = 3*np.pi/2
        nturns = 2
        rmax = 2*1e-2
        sideL1 = integrate.quad(lambda x: rmax*np.sqrt(1+x**2), theta0, theta0*(1+nturns))[0]
        sideL2 = 1/sideL1
        RESt = int(np.ceil(sideL1*RES+1))
        tdistv = np.linspace(0,sideL1,RESt)
        tv = []
        for tdist in tdistv.tolist():
            tt = fsolve(lambda x: (0.5*rmax*(x*np.sqrt(1+x**2)+np.arcsinh(x)))-\
                                   0.5*rmax*(theta0*np.sqrt(1+theta0**2)+np.arcsinh(theta0))-\
                                   tdist,theta0*(1+nturns/2))
            tv.append(tt)
        tv = np.array(tv)    
        RESh = int(np.ceil(sideL2*RES+1))
        heightv = np.linspace(0,sideL2,RESh)[:,np.newaxis]
        heightv = np.tile(heightv,[RESt,1])
        heightv = heightv.flatten('F')[:,np.newaxis]
        tv = np.repeat(tv,RESh)[:,np.newaxis]
        X=np.concatenate([rmax*tv*np.cos(tv), heightv, rmax*tv*np.sin(tv)], axis=1)
        np.random.seed(42)
        noise = 0.05
        X = X+noise*np.random.uniform(0,1,[X.shape[0],3]);
        labelsMat = np.concatenate([tv, X[:,[1]]], axis=1)
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
        labelsMat = np.concatenate([np.mod(thetav,2*np.pi), phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat
    
    def flattorus4d(self, ar=4, RES=100):
        sideLx=np.sqrt(ar)
        sideLy=1/sideLx
        Rout = sideLx/(2*np.pi)
        Rin = sideLy/(2*np.pi)
        RESx=sideLx*RES+1
        RESy=sideLy*RES+1
        x=np.linspace(0,sideLx,RESx)[:-1] # remove 2pi
        y=np.linspace(0,sideLy,RESy)[:-1] # remove 2pi
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,np.newaxis]/Rout
        yv = yv.flatten('F')[:,np.newaxis]/Rin
        X=np.concatenate([Rout*np.cos(xv), Rout*np.sin(xv), Rin*np.cos(yv), Rin*np.sin(yv)], axis=1)
        labelsMat = np.concatenate([xv, yv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat
    
    def curvedtorus3d(self, n=10000):
        Rmax=0.25;
        rmax=1/(4*(np.pi**2)*Rmax);
        X = []
        thetav = []
        phiv = []
        np.random.seed(42)
        k = 0
        while k < n:
            rU = np.random.uniform(0,1,3)
            theta = 2*np.pi*rU[0]
            phi = 2*np.pi*rU[1]
            if rU[2] <= (Rmax + rmax*np.cos(theta))/(Rmax + rmax):
                thetav.append(theta)
                phiv.append(phi)
                k = k + 1
        
        thetav = np.array(thetav)[:,np.newaxis]
        phiv = np.array(phiv)[:,np.newaxis]
        X = np.concatenate([(Rmax+rmax*np.cos(thetav))*np.cos(phiv),
                             (Rmax+rmax*np.cos(thetav))*np.sin(phiv),
                             rmax*np.sin(thetav)], axis=1)
        labelsMat = np.concatenate([thetav, phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat
    
    def kleinbottle4d(self, ar=4, RES=100):
        sideLx=np.sqrt(ar)
        sideLy=1/sideLx
        Rout = sideLx/(2*np.pi)
        Rin = sideLy/(2*np.pi)
        RESx=sideLx*RES+1
        RESy=sideLy*RES+1
        x=np.linspace(0,sideLx,RESx)[:-1] # remove 2pi
        y=np.linspace(0,sideLy,RESy)[:-1] # remove 2pi
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,np.newaxis]/Rout
        yv = yv.flatten('F')[:,np.newaxis]/Rin
        X=np.concatenate([(Rout+Rin*np.cos(yv))*np.cos(xv), (Rout+Rin*np.cos(yv))*np.sin(xv),
                          Rin*np.sin(yv)*np.cos(xv/2), Rin*np.sin(yv)*np.sin(xv/2)], axis=1)
        labelsMat = np.concatenate([xv, yv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat
    
    def mobiusstrip3d(self, ar=4, RES=90):
        sideLx=np.sqrt(ar)
        sideLy=1/sideLx
        Rmax = sideLx/(2*np.pi)
        RESx=sideLx*RES+1+50
        RESy=sideLy*RES+1
        x=np.linspace(0,sideLx,RESx)[:-1] #remove 2pi
        y=np.linspace(-sideLy/2,sideLy/2,RESy)
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,np.newaxis]/Rmax
        yv = yv.flatten('F')[:,np.newaxis]
        X=np.concatenate([(1+0.5*yv*np.cos(0.5*xv))*np.cos(xv),
                         (1+0.5*yv*np.cos(0.5*xv))*np.sin(xv),
                         0.5*yv*np.sin(0.5*xv)], axis=1)   
        labelsMat = np.concatenate([xv, yv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat