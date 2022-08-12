import scipy.sparse as sp
import scipy.special as spc

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import umap.umap_ as um

from copy import deepcopy
import pickle as pc
import matplotlib as mpl

import dill as pc
import datetime as dt
import os

plt.rcParams['figure.figsize'] = [10,10]

np.random.seed(54321)


NETWPROPS_ELEM = ["en", "magn", "staggmagn", "numconn", "accrat_spins", "accrat_conns", "spins", "clusternums", "clustering", "density","sizedistr","adj"]
NETWPROPS_AVG = ["heatcap", "spins","adj"]
DATA_AVG = {"spins": lambda calc,tnum,dim: {"mean": np.sum(calc["spins"][tnum]["mean"]**2)/dim, "std": np.sqrt(np.sum(np.abs(2*calc["spins"][tnum]["mean"]*calc["spins"][tnum]["std"])**2))/dim**2}, "adj3": lambda calc,tnum,dim: {"mean": (np.sum(calc["adj3"][tnum]["mean"]**2)-dim)/(dim*(dim-1)),"std": 0}, "heatcap": lambda calc,tnum,dim: {"mean": calc["en"][tnum]["std"]**2 * dim/ calc["temp"][tnum]**2, "std": 0}}

PLOTNETWPROPS = {"eigvals": lambda g, ax, dim: ax.hist(nx.adjacency_spectrum(g),10*dim), "laplacian": lambda g, ax, dim: ax.hist(nx.laplacian_spectrum(g),dim)} 
PLOTMODELPROPS = {"spindistr": lambda cast, ax: ax.hist(cast.spin,cast.dim), "conndistr": lambda cast, ax: ax.hist(cast.nconnlist(),np.linspace(0,20,20)-.5,density=True,histtype="step",color="black"), "connendistr": lambda cast,ax: ax.hist([cast.frustr(*i) for i in np.array(sum([[[i,j] for j in cast.neigh[i]] for i in range(cast.dim)],[]))], np.linspace(-10,10,1000))}




# Creates the (sparse) adjacency matrix for a 1D chain with N nodes
def onedneigh(N):
    tmp = []
    for i in range(N):
        tmp.append([(i-1)%N, (i+1)%N])
    return tmp

# Creates the (sparse) adjacency matrix for a randomly connected matrix with N nodes.
# where 'minnum' is the minimal amount of connections per node
# - currently it is exact only for values 0,1.
def rand(N, minnum,maxnum=np.infty,rettmp=False):
    assert minnum <= N, "Minimal connections required exceeds all-to-all connection"
    assert minnum*N <= maxnum
    neigh = [[] for i in range(N)]
    tmp = []
  
    if minnum==1:
        tmp=[list(np.arange(0,N)) for i in range(N)]
        
        for i in range(N):
            tmp[i].remove(i)
        return tmp
    
  
    elif minnum==-1:
        tmp = rand(N,maxnum/N,maxnum,rettmp=True)
        del tmp[np.random.randint(len(tmp))]
         
    else:
        u = list(np.arange(N))
        while len(tmp) < minnum*N*(N)/2:
            try:
                c = list(np.random.choice(u,size=2,replace=False))
            except:
                tmp = rand(N,minnum,maxnum,rettmp=True)
                break
                
            if c==u or c==reversed(u):
                tmp = rand(N,minnum,maxnum,rettmp=True)
                break
                
            if not (c in tmp or list(reversed(c)) in tmp):
                tmp.append(c)    
                if np.sum(tmp == c[0]) >= maxnum:
                    u.remove(c[0])
                if np.sum(tmp == c[1]) >= maxnum:
                    u.remove(c[1])
    
    if rettmp:
        return tmp
      
    for conn in tmp:
        neigh[conn[0]].append(conn[1])
        neigh[conn[1]].append(conn[0])
    return neigh
    
    
# Creates dense matrix alpha_ij = Uniform(-width/2, width/2)
#def alphgen(N,width):
#    assert width <= np.pi
#    return np.random.rand(N,N)*width
  
#The data type for a XY-rotor, storing its shape, spins, temperature, etc.      
class XY:
    
    XS = np.linspace(0,2*np.pi,1000, endpoint=False)
    
    JFUNS  = lambda rot: {"trivial": lambda i,j: 1, "phys": lambda i,j: 1, "kac": lambda i,j: (rot.dim/2-1/2)/(1+rot.nconnlist()[i]+rot.nconnlist()[j])}
    JFUNSMATR  = {"trivial": lambda nconnmatr, dim: np.ones((dim,dim)), "phys": lambda nconnmatr, dim: np.ones((dim,dim)), "kac": lambda nconnmatr, dim: 1/(1+nconnmatr.T+nconnmatr)}
    
    def __init__(self, neigh, jj, t, spin = None, alpha=None, jfun="trivial", calcdepth=1, conn=[np.infty,"nonfixed"]):
        self.neigh = deepcopy(neigh)
        self.jj = deepcopy(jj) # -> switch to sparse array instead of matrix
        
        self.beta = 1/t
        self.dim = len(neigh)
        self.depth = calcdepth
    
        if type(alpha) != np.ndarray:
            self.alpha = np.zeros((self.dim, self.dim))
        else: self.alpha = deepcopy(alpha)

        if type(spin) != np.ndarray:
            self.spin = 2*np.pi*np.random.rand(self.dim)
        else: self.spin = deepcopy(spin)
        
        if sum([len(neigh[i]) for i in range(self.dim)])==self.dim*conn[0]-2:
            for i in range(self.dim):
                if len(neigh[i])==self.dim-1:
                    continue
                for j in range(self.dim):
                    if j not in neigh[i] and j!=i and len(neigh[i]) < conn[0]:
                        self.freenodes = np.array([i,j])
                        break
                else:
                    continue
                break

            
        else: self.freenodes = None
            
        
        self.CMAPS = {"spins": ["twilight", lambda : self.spin/(2*np.pi), lambda : 2*np.pi], "frustr": ["viridis", lambda : .5+np.array([np.sum(self.frustr(i)) for i in range(self.dim)])/2/3+0*np.amax(np.array([np.sum(self.jhand(j,self.neigh[j])) for j in range(self.dim)]), initial=1), lambda : np.amin(np.array([-np.sum([self.jhand(j,self.neigh[j])]) for j in range(self.dim)]))], "disparity": ["viridis", lambda: np.sum(np.multiply(XY.denseneigh(self.neigh), self.jmatr())**2, axis=0)/(np.sum(np.multiply(XY.denseneigh(self.neigh),self.jmatr(),axis=0)**2)), lambda : 1]}
            
        self.EDGELABS = {"frustr": ["viridis", lambda edges: (1+np.array([self.frustr(edg[0],edg[1])/self.jhand(edg[0],edg[1]) for edg in edges])/3)/2, 1], "pearsoncorr": ["viridis", self.pearsoncorrfun, 1]} 
        
        self.conn = conn[1]
        self.snaps = {"sweepnum": [], "neigh": [], "spin": []}
        self.jfun=XY.JFUNS(self)[jfun]
        self.jfunname=jfun
    
    def pearsoncorrfun(self,edges):
        ii = np.multiply(XY.denseneigh(self.neigh), self.jmatr())
        diff = ii-np.sum(ii,axis=0)/self.dim
            
        sq = np.sqrt(np.sum(diff**2,axis=0))
        diffsq = diff@(diff.T)
            
            
        return (1+np.array([diffsq[edg[0],edg[1]]/(sq[edg[0]]*sq[edg[1]]) for edg in edges]))/2                    

    def takesnap(self, sweepnum):
        self.snaps["sweepnum"].append(sweepnum)
        self.snaps["neigh"].append(deepcopy(self.neigh))
        self.snaps["spin"].append(deepcopy(self.spin))
    
    # The 'frustration' or energy of i-th node.
    def frustr(self, i, j=None, spineval=None):
        
        if j==None:
            neighb = self.neigh[i]
        elif j==[]:
            return [0]
        else:
            neighb=j
            
        if spineval==None:
            spineval=self.spin[i]
        elif neighb==[]:
            return 0
             
        return -self.jhand(i,neighb)*np.cos(self.spinang(spineval,self.spin[neighb])-self.alpha[i,neighb])
        
    def nconnlist(self):
        return np.array([len(self.neigh[i]) for i in range(self.dim)])   
        
    def pdf(self,i,spineval):
        js = self.neigh[i]
        if js==[]:
            return [1]*len(spineval)
        return np.exp(self.beta*np.sum(self.jhand(i,js)*np.cos(self.spinang(spineval[:,None],self.spin[js])-self.alpha[i,js]),axis=1) - self.beta*np.sum(self.jhand(i,js)))
 
    def pos(self):
        
        l = np.array(sum([[[i,j] for j in self.neigh[i]] for i in range(self.dim)],[]))
        if l.size==0: 
            return {0: [0,0]}

        

     #   if self.dim==2:
     #       return {0: [-1,0], 1: [1,0]}
        val = np.max(self.jmatr()[l[:,0],l[:,1]])
        
        
        h = np.zeros((self.dim,self.dim))
        for i,j in l:
            h[i,j]=(self.frustr(i,j)/val)/2
        h+=1/2
        
        if np.linalg.matrix_rank(h-.5) == 2:
            dim = len(h)
            h-=.5
            coors = np.zeros((dim,2))
            for i in range(dim):
                count=0
                for n in range(1,dim):
                    ph = count*2*np.pi/len(h[i][h[i]!=0])
                    if h[i,n] != 0 and not coors[n].any():
                        coors[n,0] = coors[i,0]+h[i,n]*np.cos(count*ph)
                        coors[n,1] = coors[i,1]+h[i,n]*np.sin(count*ph)
                        count+=1
            
            return {i : coors[i] for i in range(dim)}
        
        red = um.UMAP(random_state=42,metric="precomputed").fit_transform(h)
        
        return {i : red[i] for i in range(self.dim)}
        
    def spinang(self,thetai,thetaj):
        return np.pi-np.abs(np.abs(thetai-thetaj)-np.pi)
        
    def updatespins2(self):
        updnode = np.random.randint(0,self.dim)
        js = self.neigh[updnode]
            
        if js==[]:
            arg = np.array([1]*len(XY.XS))
        else: 
            arg = np.sum(self.jhand(updnode,js)*np.cos(self.spinang(XY.XS[:,None],self.spin[js])-self.alpha[updnode,js]),axis=1)
        p = np.exp(self.beta*(arg-np.max(arg)))
        
        oldspin = self.spin[updnode]
        self.spin[updnode] = np.random.choice(XY.XS,p=p/np.sum(p))
        
        return np.abs(self.spin[updnode]-oldspin)
    
    def jhand(self, i,j):
        return self.jj[i,j]*self.jfun(i,j)
    
    def jmatr(self):
        ind = np.array([np.arange(self.dim)]*self.dim)
    
        return np.multiply(self.jj,self.jfun(ind.T,ind))
    
    # Updates randomly N nodes (and edges if needed), with N the amount of nodes.
    def sweep(self,fixedconn,num=1,maxconn=["global",np.infty]):
        accrat = {"accrat_spins": 0, "accrat_conns": 0}
        for i in range(self.dim*num):
            accrat["accrat_spins"] += self.updatespins2()
            if not fixedconn:
                accrat["accrat_conns"] += self.updateconnections(maxconn)
        return accrat

    def sett(self,t):
        self.beta = 1/t

    # Returns a dense form of the sparse adjacency matrix.
    @staticmethod 
    def denseneigh(neigh):
        dim = len(neigh)
        out = np.zeros((dim,dim))
        for i in range(dim):
            for j in neigh[i]:
                out[i,j]=1

        return out
        
    # A random node is picked. This node is given a potential new spin that is uniformly picked from 0 to 2*pi.
    # From a Boltzmann distribution it is decided whether or not that nodes spin changes to this potential spin.
    # To be able to count the acception rate 1 is returned upon change and 0 otherwise
    def updatespins(self): #could be broke
        updnode = np.random.randint(0,self.dim)#

        oldspin = self.spin[updnode]
        oldfrustr = np.sum(self.frustr(updnode))
        self.spin[updnode] = 2*np.pi*np.random.rand()

        if not self.acceptchange(np.sum(self.frustr(updnode))-oldfrustr):
            self.spin[updnode]=oldspin
            return 0
        return 1

    # Two nodes are randomly picked. If udsonconnected, the potential change is a connection. If connected it is in reverse. 
    # Again from a Boltzmann distribution it is decided if the connection should be made/broke or not.
    # To be able to count the acception rate 1 is returned upon change and 0 otherwise 
   
    
   
    def updateconnections(self, maxconn):
      
        if self.conn=="nonfixed":
            bool = True
            while bool:
                i,j = np.random.randint(len(self.neigh), size=2)
                bool = i==j
    
            dE = self.frustr(i,j) 
            
            if i in self.neigh[j]:
                if self.acceptchange(-dE):
                    self.remove(i,j)
                    return 1
            else:
                if maxconn[0]=="local":
                    maxconnok = len(self.neigh[i]) < maxconn[1] and len(self.neigh[j]) < maxconn[1]
                elif maxconn[0]=="global":
                    maxconnok = self.connnum() < maxconn[1]
                if self.acceptchange(dE) and maxconnok:
                    self.add(i,j)
                    return 1
            return 0
        if self.conn=="fixed":
            if maxconn[0]=="local":
                boolfrom=True
                while boolfrom:
                    fromm = np.random.randint(len(self.neigh), size=2)
                    boolfrom = not fromm[0] in self.neigh[fromm[1]] or fromm[0]==fromm[1] 
            
                if isinstance(self.freenodes, type(None)):
                    nds = np.arange(self.dim)[self.nconnlist()<maxconn[1]]
                    
                    nds = np.hstack((fromm, nds))
                    boolto = True
                    while boolto:
                        to = np.random.choice(nds,size=2,replace=False)  
                        boolto = to[0] in self.neigh[to[1]] or to[0]==to[1]
                
                
                else:
                    
                    nds = [self.freenodes,fromm]
                    to = nds[np.random.randint(2)]
                    boolto = True
                   
            else:
                bool = True
                while bool:
                    l = np.random.randint(len(self.neigh), size=(2,2))
                    ex = np.array([l[0,0] in self.neigh[l[0,1]], l[1,0] in self.neigh[l[1,1]]])
                    
                    bool = np.sum(l[:,0]==l[:,1]) or np.array_equal(l[0],l[1]) or np.array_equal(l[0],np.flip(l[1])) or not (ex[0] ^ ex[1])
                    
                to, fromm = l[ex.argsort()]    
                
            dE = self.frustr(*to) - self.frustr(*fromm)
           
            if self.acceptchange(dE):
                self.remove(*fromm)
                self.add(*to)
                
                if not isinstance(self.freenodes, type(None)) and not np.array_equal(fromm,to):
                    self.freenodes = fromm
                    
                return 1
            return 0
            
        if self.conn == "fixed2":
            bool = True
            while bool:
                i1 = np.random.randint(self.dim,size=2)
                i2 = np.array([np.random.choice(self.neigh[j]) for j in i1])
                to = np.hstack((i1[:,None],np.flip(i2)[:,None]))
                fromm = np.hstack((i1[:,None],i2[:,None]))
                bool = np.array_equal(fromm[0],fromm[1]) or np.array_equal(fromm[0],np.flip(fromm[1])) or np.array_equal(to,fromm) or np.any(to[:,0]==to[:,1]) or to[0,0] in self.neigh[to[0,1]] or to[1,0] in self.neigh[to[1,1]] 
            
            dE = self.frustr(*to[0])+self.frustr(*to[1]) - (self.frustr(*fromm[0])+self.frustr(*fromm[1]))
            if self.acceptchange(dE):
                self.remove(*fromm[0])
                self.remove(*fromm[1])
                self.add(*to[0])
                self.add(*to[1])
                return 2
            return 0
                
    # The number of connections between nodes
    def connnum(self):
        return sum([len(l) for l in self.neigh])/2
    
    
    def remove(self,i,j): 
        self.neigh[i].remove(j)
        self.neigh[j].remove(i)
        
    def add(self,i,j):
        self.neigh[i].append(j)
        self.neigh[j].append(i)

    # All data you may need concerning the system as a whole.
    def getdata(self):
        #energy
        en=0
        for i in range(self.dim):
            en += np.sum(self.frustr(i))
        
        #staggered magnetisation - defined in such a way that for alpha_ij=pi, 
        #maximal staggered magnetisation corresponds to minimal energy
        l = np.arange(self.dim).tolist()
        i=0
        while i < len(l):
            for j in self.neigh[l[i]]:
                if j in l:
                    l.remove(j)
            i+=1
        l = np.array(l)
        
        neighs = []
        for i in l:
            neighs.append(self.neigh[i])
        neighs = list(set((sum(neighs,[])))) 
            
        frustr = False    
        for i in neighs:
            if i in l:
                frustr = True
        if len(neighs)+len(l) != self.dim:
            frustr = True
        
        magnset = lambda arr: np.array([np.sum(np.cos(self.spin[arr])), np.sum(np.sin(self.spin[arr]))])
        staggmagn = 0 
        if not frustr:
            staggmagn = np.linalg.norm(magnset(l) - magnset(neighs))
            
        magn = np.linalg.norm(magnset(np.arange(self.dim).tolist()))
                
        
        ii = XY.denseneigh(self.neigh)
        iisq = ii@ii 
            
        sizedistr = [len(_) for _ in nx.connected_components(nx.from_scipy_sparse_matrix(sp.csr_matrix(XY.denseneigh(self.neigh))))]
        
        
        return {"en": (en/2)/self.dim, "magn": magn/self.dim,"staggmagn": staggmagn/self.dim, "numconn": len(sum(self.neigh,[]))/(self.dim*(self.dim-1)), "spins": [np.cos(self.spin),np.sin(self.spin)], "clusternums": len(sizedistr), "clustering": (np.trace(iisq@ii)/(np.sum(iisq)-np.trace(iisq))) if np.sum(iisq)-np.trace(iisq) else 0, "density": np.sum(ii)/(self.dim*(self.dim-1)), "sizedistr": sizedistr, "adj": ii,"adj2": 2*(ii-.5),"adj3": ii-np.mean(ii)}
            
    # Boltzmann RNG 
    def acceptchange(self,dE):
        p = np.exp(-self.beta*dE)
        return p > np.random.rand()
        
    
# Runs one full calculation. If the system is started from equilibrium, first an amount of sweeps are done to get it there.
# The same amount of sweeps are ran to collect data. Note that for unfixed connections for each temperature a new rotor is made,
# while for fixed connections the temperature is changed at each model. 

# It returns all data that you may want concerning averaged data (of the system in equilibrium), and the non-averaged data for a dynamical system.
# It also returns the 'XY'-instances at certain moments in time (at certain sweep moments), intended to make plots of the graph and if you want to know 
# a property of the rotor at one specific moment in time. 

# This function has a handler to execute it as a script: "calc.py".

def runcalculation(neigh,j,ts,fixedconn,sweeps,savenums,savedynamics=False, equilibrium=True, alpha=None, alphwidth=0,spin=None,maxconn=["global",np.infty], jfun="trivial", calcdepth=1, conn=[np.infty,"nonfixed"]):
    if fixedconn:
        rot = XY(neigh,j,ts[0],spin,alpha,jfun,calcdepth,[maxconn[1],conn])
   
    
    calc = {arg: [None]*len(ts) for arg in LABCALC}
    dictgen = lambda labs, *args: {arg: [] for arg in (list(args)+labs)}
    calc["temp"] = ts
    
    dyn = [] 
    snaps = []
    for n in range(len(ts)):        
        if fixedconn:
            rot.sett(ts[n])
        else:
            rot = XY(neigh,j,ts[n],spin,alpha,jfun,calcdepth,[maxconn[1],conn])
   
        sweepdata = dictgen(NETWPROPS_ELEM)
         
        if -1 in savenums:
            rot.takesnap(-1)
        
        if equilibrium:
            rot.sweep(fixedconn,sweeps,maxconn=maxconn)
        for i in range(sweeps):
            accrat = rot.sweep(fixedconn,maxconn=maxconn)
            data = {**rot.getdata(), **accrat}
            
            for key in NETWPROPS_ELEM:
                sweepdata[key].append(data[key])
            
            if i in savenums:
                rot.takesnap(i)
        
        for key in NETWPROPS_ELEM:
            if key != "sizedistr":
                calc[key][n] = {"mean": np.mean(np.array(sweepdata[key]),axis=0), "std": np.std(np.array(sweepdata[key]),axis=0)}
        
        if savenums != []:
            snaps.append(deepcopy(rot))
        
        #Additional data calculated from averaged data in calc. 'spins' does not really fit: in reality this abuses the 'mean' to average over time instead of nodes; and puts a vector in 'calc', which is then overwritten manually. So: it is dynamics which cannot have the default np.mean(...) to compute the mean.
     
        for key in NETWPROPS_AVG:
            calc[key][n] = DATA_AVG[key](calc, n, rot.dim)
     
        del sweepdata["spins"]
        del sweepdata["adj"]
        
        if savedynamics:
            dyn.append(sweepdata)
        print(ts[n], "done")
    print("calcdone")
            
    return [calc,sweeps,alphwidth],snaps, [dyn, equilibrium, ts,rot.dim]



#func needs update

# If you want to repeat 'runcalculation'. The means/stds are added together accordingly.
# It is useful to get a variance on the specific heat, which it lacked because it itself 
# depends on the variance of energy.
def repeatcalc(n,dim,neighfun,j,ts,fixedconn,sweeps,alphafun=lambda n:None,spinfun=lambda n:None,posfun=lambda n:None):  #functions need only depend on dim
    calclist = []
    for i in range(n):
        calc = runcalculation(neighfun(dim),j(dim),ts,fixedconn,sweeps,[],alphafun(dim),spinfun(dim),posfun(dim))[0][0]
        calclist.append(calc)
        with open("calclistbackup.txt", "wb") as f:
            pc.dump(calclist, f)
    
    calcrepeat = {"temp": ts, "en": [], "magn": [], "staggmagn": [], "heatcap": [], "numconn": [], "accrat_spins": [], "accrat_conns": []}
    
    for t in range(len(calc["temp"])):
        for key in calc.keys():
            if key != "temp" and key != "heatcap":
            #teh stds are added together even though they are not fully independent in case of fixed connection! 
                calcrepeat[key].append({"mean": np.mean(np.array([calclist[i][key][t]["mean"] for i in range(n)])), "std": np.sqrt(np.sum(np.array([calclist[i][key][t]["std"] for i in range(n)])**2))})
        calcrepeat["heatcap"].append({"mean": np.mean(np.array([calclist[i][key][t]["mean"] for i in range(n)])), "std": np.std(np.array([calclist[i][key][t]["mean"] for i in range(n)]))})
    return [calcrepeat, sweeps*n] 






 
def fixslash(arg):
    if arg == '' or arg[-1] == "/":
        return arg
    return arg+'/'

def corrs(sweepdata,tnum,dim,restr):
    l = len(sweepdata["spins"])
    out = np.tile(np.expand_dims(np.transpose(sweepdata["spins"], axes=(0,2,1)),1),(1,dim,1,1))
    return {"mean": np.sum(np.einsum('ijkl,ijkl->ijk',out,np.transpose(out,axes=(0,2,1,3))),axis=0)/l, "std":0}
    

LABCALC = ["temp"]+NETWPROPS_ELEM+NETWPROPS_AVG+["corr"]

LABSTRINGS = ["$T$", "Energy per spin $E/N$", "Magnetisation per spin $|M|/N$", "Antiferromagnetic staggered magnetisation per spin $|M_S|/N$",  "$\overline{z}/(N-1)$", "Acception rate of spins", "Acception rate of connections", "","Amount of disjoint subgraphs", "Clustering", "Density: average of $J_{ij}$", "sizedistr", "Heat capacity per spin $C/N$", "Edward-Anderson parameter $q$","correlations"]

while len(LABSTRINGS) < len(NETWPROPS_ELEM) + len(NETWPROPS_AVG):
    LABSTRINGS.append("<NO LABEL>")
LABS = dict(zip(LABCALC, LABSTRINGS))

# plots the figure, either to screen, either to file.
def plotout(fig, tit, savedir,loc):

    
    axs=fig.get_axes()
    
    
    
    # to change the figures title, range, etc.
    # ---------------------
    
    
    
    # ---------------------
    
    d=4.65
    x=d+.5
    y=x-.4
    
    xlarge = x*(4/3)
    ylarge = y*(4/3)
    fig.set_size_inches(x,y)
    fig.tight_layout()
    
    if savedir==None:
        fig.show()
    else:
        fig.savefig(fixslash(loc)+dirtoday()+fixslash(savedir)+tit+".pdf")
        plt.close()

# saves/retrieves data to/from some kind of unreadable format specified by 'pickle' package.
# used to store calculation results and later access it for plots.
def saverawdata(dat,savedir,loc="",fname="rawdata.txt"):
    if savedir != None:
        with open(loc+"/"+dirtoday()+savedir+"/"+fname, "wb") as f:
            pc.dump(dat, f, protocol=pc.HIGHEST_PROTOCOL)

def getrawdata(savefile):
    if savefile != None:
        with open(savefile, "rb") as f:
            return pc.load(f)
    else:
        return None
    
    
def dirtoday():
    return dt.datetime.today().strftime('%d%m%y')+"/"

# not to waste time computing something while at the end the directory to store is invalid.
def direxists(loc, savedir):
    if savedir == None:
        return True
    return os.path.isdir(loc+"/"+dirtoday()+savedir)

def boolpars(yn):
    if yn == "y":
        return True
    if yn == "n":
        return False
    return None


# This function plots everything a plot is compared to ('grey lines'). (Since it is a help function to plot, the code is less clean)
def plotpred(ax, calc, ykey, xkey, errbars, cont=True):
    
    numconny=np.array([0.75992318, 0.75976456, 0.75943497, 0.75890075, 0.75816272, 0.75721009, 0.75601899, 0.754559  , 0.75280233, 0.7507314, 0.74834251, 0.74564592, 0.74266355, 0.73942569, 0.73596775, 0.73232734, 0.72854205, 0.72464787, 0.72067807, 0.71666269, 0.7126282 , 0.70859753, 0.70459019, 0.70062249, 0.69670787, 0.69285718, 0.689079  , 0.68537998, 0.68176504, 0.67823772, 0.67480033, 0.67145419, 0.66819976, 0.66503688, 0.66196477, 0.65898227, 0.65608782, 0.65327962, 0.65055567, 0.64791378, 0.64535168, 0.64286704, 0.64045747, 0.63812057, 0.63585395, 0.63365525, 0.63152212, 0.62945228, 0.62744349, 0.62549357])
    
    tmpeig = np.array([0.49683263,  0.56627066,  0.78409214,  1.12330072,  1.17182916,  1.36690236,1.51190451,  1.66811595,  1.75913622,  1.81200581,  1.83673804,  1.90692596,1.92878876,  2.03122055,  2.12185228,  2.15404642,  2.33035065,  2.349759,2.41818672,  2.52354822,  2.61412724,  2.75509964,  2.77116244,  2.85672821,2.87755621,  3.08412701,  3.26219177,  3.31636588,  3.39593929,  3.62518799,3.6744395,  3.70227343, 3.83896051,  3.88101537,  3.95318424,  4.03251349,4.11419685,  4.18169316,  4.31733948,  4.38471294,  4.48538475,  4.51880976,4.75502272,  4.77652916,  5.05273214,  5.10187597,  5.21356357,  5.37234829,5.54416827,  5.58175462,  5.64388178,  5.78875806,  5.96755123,  6.02216076,6.1860728,   6.30584632,  6.41160874,  6.48856995,  6.52156158,  6.57391703,6.77528748,  6.85366225,  6.99302297,  7.22554338,  7.25791507,  7.43604378,7.63611838, 7.78809955,  8.29815871,  8.45615958, 8.56053962,  8.69481737,8.76210947,  8.84110969,  8.95723092,  9.01220742,  9.11757728,  9.38320221,9.55956317,  9.61377431,  9.95354917, 10.1003358,  10.27880343, 10.65576998,10.82121669, 10.91013504, 11.24255791, 11.31659593, 11.37575687, 11.41203494,11.90156993, 12.0303462,  12.04264407, 12.62948036, 12.79700663, 13.27153311,13.73214365, 13.79445672, 15.18069496])  #example
    
    assert xkey == "temp"
    n=100
    
    if ykey=='en' or ykey=='heatcap':
        
        if not cont:
            temp = calc[0]["temp"]
            temp = np.array([temp,np.ones(len(temp))])
            numconn = (n-1)*np.array([[float(calc[0]["numconn"][i][j] or 0) for i in range(len(calc[0]["numconn"]))] for j in ["mean","std"]])/2
            temp2=deepcopy(temp)
            temp2[1,:]=0
            
            
            
        else:
            temp = np.linspace(calc[0]["temp"][0], calc[0]["temp"][-1]*1,100)
            temp2 = temp  
            numconn = (n-1)*np.array([float(calc[0]["numconn"][i]["mean"] or 0) for i in range(len(calc[0]["numconn"]))])/2
            
        if ykey=='en':
            #pred1 = -2.5*spc.iv(1,1/temp2)/spc.iv(0,1/temp2) #for the 1D plot
            
            pred1 = -numconn+(numconn-2/100)*temp/2-np.array([np.sum((tmpeig/8)*(1-np.exp(-b*tmpeig/8)/(np.sqrt(np.pi*b*tmpeig/8)*spc.erfc(np.sqrt(b*tmpeig/8))))) for b in 1/temp2])
            
            pred2=-numconn+numconn*temp/2-temp2**2*4*np.sum(1/tmpeig)/100  #Taylor expansion to 2nd order
            
        if ykey=='heatcap':
            #mu=spc.iv(1,1/temp2)/spc.iv(0,1/temp2)  #for the 1D plot
            #pred1 = (1/temp2)**2*(1-mu*temp2-mu**2)
            
            pred0 = numconn/2
            
            
        ax.plot(temp if cont else temp[0], pred0 if cont else pred0[0], '--', color="grey") #dash
        ax.plot(temp if cont else temp[0], pred1 if cont else pred1[0], color="grey")
        #ax.plot(temp if cont else temp[0], pred1 if cont else pred1[0], '-',dashes=[8, 4, 2, 4, 2, 4], color="grey") #dash dot dot
        
        if errbars and not cont:
            ax.errorbar(temp[0],pred0[0],yerr=2*pred0[1],fmt='none',color="grey")
            ax.errorbar(temp[0],pred1[0],yerr=2*pred1[1],fmt='none',color="grey")
            
    if ykey=="numconn":
        ax.plot(np.linspace(.01,.8),numconny,color="grey")
        ax.set_xlim([None,.8])
        
        

# plot general averaged properties of the system.
# it can be executed as a script through plotprop.py
def plotprop(calc,ykey, xkey="temp", title="", errbars = True, savedir=None,loc="", writeplot=[True,False],label=None,tit=None):
       
    yrestruc = {j: np.array([float(calc[0][ykey][i][j] or 0) for i in range(len(calc[0][ykey]))]) for j in ["mean","std"]}
    fig, ax = plt.subplots()   
    label = label if label!=None else '$alpha$ = '+str(round(calc[2],2))
   
    ax.plot(calc[0][xkey],yrestruc["mean"],'o',label=label, color='black')
    
    if errbars:
        ax.errorbar(calc[0][xkey],yrestruc["mean"], yerr = 2*yrestruc["std"],fmt='none',color="black")
    ax.set_xlabel(LABS[xkey])
    
    tit = tit if tit != None else title+" ("+str(calc[1])+" sweeps; alpha="+str(round(calc[2],2))+")"
    
    if writeplot[0]:
        plotout(fig, ykey+" - "+tit, savedir,loc)
    if writeplot[1]:   
        saverawdata(ax, savedir, loc, fname=ykey+" - "+tit+".txt")
    
# plot snapshots of the system at certain moments
# it can be executed as a script through plotsnaps.py

def plotgraph(snaps,title="",savedir=None,loc="",addgraphs=None):
    for rot in snaps:
        tformat = str(round(1/rot.beta,5))
        os.mkdir(fixslash(loc)+dirtoday()+fixslash(savedir)+tformat)        
        for i in range(4,len(rot.snaps["sweepnum"])):
            fig,ax = plt.subplots()
            
            cast = XY(rot.snaps["neigh"][i], rot.jj, 1/rot.beta, rot.snaps["spin"][i], rot.alpha, rot.jfunname)
            if addgraphs in PLOTMODELPROPS.keys():
                PLOTMODELPROPS[addgraphs](cast, ax)
            elif addgraphs in PLOTNETWPROPS.keys():
                g = nx.from_numpy_matrix(XY.denseneigh(rot.snaps["neigh"][i]))    
                PLOTNETWPROPS[addgraphs](g, ax, cast.dim//2)
        
            if addgraphs == "conndistr":
         
                p=250/spc.comb(99,2)
                fun = lambda k: spc.comb(99,k)*p**k*(1-p)**(99-k)
                xr= np.linspace(0,10.5,num=1000)
         
                ax.plot(xr,fun(xr),color="grey")
        
            dat = cast.getdata()
            
            tit = "System snapshots at T = "+tformat+". After "+str(rot.snaps["sweepnum"][i])+" sweeps."
            plotout(fig,title+" - "+tit[20:], fixslash(savedir)+tformat, loc)
    
def printtmp(xy):
    
    a = XY.denseneigh(xy.neigh)
    
    A = (np.diag(np.sum(a,axis=0))-a)
    l,v = np.linalg.eig(A)
    l=l.real
    v=v.real
    
    aa = np.zeros((100,100))
    for i in np.argwhere(l>1e-5):
        aa+=(1/l[i])*np.outer(v[:,i],v[:,i])
        
    l = np.tile(np.expand_dims(aa, axis=(1,3)), (1,100,1,100))
    l = -(l + np.transpose(l, axes=(1,0,3,2)) - np.transpose(l, axes=(1,0,2,3)) - np.transpose(l, axes=(0,1,3,2))) 
    l2 = np.einsum('ij,mn->ijmn',a*xy.alpha,a*xy.alpha)
    l=np.multiply(l,l2)
    l=np.vstack(l)
    l=np.transpose(l,axes=(2,1,0))
    l=np.vstack(l)
   
    l=l[~np.all(l == 0, axis=1)]
    l=l[:,~np.all(l == 0, axis=0)]
    
    lam=np.linalg.eigh(l)[0]
    lamnonz = lam[np.abs(lam)>1e-6]
  
    print("Eigvals of Q",lamnonz)

def plotsnaps(snaps, title="", savedir=None,loc="", nodekey="spins", edgekey=None, addgraphs=[], pos="default", printeigvals=False):
    
    maxnum=3
    for rot in snaps:
        tformat = str(round(1/rot.beta,5))
        os.mkdir(fixslash(loc)+dirtoday()+fixslash(savedir)+tformat)
        for i in range(len(rot.snaps["sweepnum"])):
             
            g = nx.from_numpy_matrix(XY.denseneigh(rot.snaps["neigh"][i]))      
            subgraphs = [g.subgraph(c).copy() for c in nx.connected_components(g)]
            
            fig = plt.figure(constrained_layout=True)
            extrcol = addgraphs != []
            
            rownum= len(subgraphs)//maxnum+bool(len(subgraphs)%maxnum)
            plt.rcParams['figure.figsize'] = [10*max(np.maximum(maxnum,len(subgraphs)), len(addgraphs)), 10*(rownum+int(extrcol))]
            subfigs = fig.subfigures(1+int(extrcol), 1, squeeze=False, height_ratios=([2]+[1]*int(extrcol))).flatten()
            
            
            axsplot = subfigs[0].subplots(rownum,1+np.minimum(maxnum,len(subgraphs)), gridspec_kw = {"width_ratios": [1]*np.minimum(maxnum,len(subgraphs))+[.1]},squeeze=False)
            
            if extrcol:
                axsgraph = subfigs[1].subplots(1, len(addgraphs), squeeze=False).flatten()
           
            for k in range(len(subgraphs)):
                
                if len(subgraphs) > 1:
                    nods = np.array(subgraphs[k].nodes)
                    subneigh = deepcopy(rot.snaps["neigh"][i])
                    
                    l=0
                    c2=0
                    while l-c2 < len(subneigh):
                        
                        if l not in nods:
                            del subneigh[l-c2]
                            for m in range(len(subneigh)):
                                for n in range(len(subneigh[m])):
                                    if subneigh[m][n] > l-c2:
                                        subneigh[m][n] -= 1
                            c2+=1
                        l+=1
                          
                    cast = XY(subneigh, rot.jj[nods[:,None],nods], 1/rot.beta, rot.snaps["spin"][i][nods], rot.alpha[nods[:,None],nods], rot.jfunname)
                    castpar = XY(rot.snaps["neigh"][i], rot.jj, 1/rot.beta, rot.snaps["spin"][i], rot.alpha, rot.jfunname)
                    
                            
                else:
                    cast = XY(rot.snaps["neigh"][i], rot.jj, 1/rot.beta, rot.snaps["spin"][i], rot.alpha, rot.jfunname)
                    castpar = cast
                    
                if printeigvals:
                    printtmp(castpar)
         
                subgraphs[k] = nx.convert_node_labels_to_integers(subgraphs[k])
                
                
                if pos=="default":
                    posdict = nx.spring_layout(subgraphs[k])
                    
                elif pos=="frustr":
                    posdict = cast.pos()
                 
                if nodekey=="arrow":
                    posarr = np.array(list(posdict.values()), dtype=float)
                    dx = np.cos(cast.spin)/50
                    dy = np.sin(cast.spin)/50
                    nx.draw_networkx_edges(subgraphs[k], posdict,ax=axsplot[k//maxnum,k%maxnum])
                    for _ in range(cast.dim):
                        axsplot[k//maxnum,k%maxnum].arrow(posarr[_,0]-dx[_], posarr[_,1]-dy[_],2*dx[_],2*dy[_], width=.01)
                        axsplot[k//maxnum,k%maxnum].text(posarr[_,0],posarr[_,1],str(_))
                    
                
                else:
                    cmap = plt.get_cmap(castpar.CMAPS[nodekey][0])
                
                    
                    if edgekey == None:
                        edgecol='k'
                        edglist = list(subgraphs[k].edges())
                    elif edgekey == "no":
                        edglist = []
                    else:
                        edges = sum([[(i,j) for j in cast.neigh[i]] for i in range(cast.dim)], [])
                        edgecol = cmap(cast.EDGELABS[edgekey][1](edges))
                        edglist = list(subgraphs[k].edges())
                    
                    
                    nx.draw_networkx(subgraphs[k],pos=posdict, node_color=cmap(cast.CMAPS[nodekey][1]()[subgraphs[k].nodes]),ax=axsplot[k//maxnum,k%maxnum],node_size=200, edgelist = edglist, edge_color=edgecol, with_labels=False)
                     cm.ScalarMappable(norm=norm, cmap=cmap)
           
                axsplot[k//maxnum,k%maxnum].set_aspect("equal")
           
            for k in range(k+1,rownum*np.minimum(len(subfigs),maxnum)):
                axsplot[k//maxnum,k%maxnum].set_aspect("equal")
           
            if nodekey != "arrow": fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin=-castpar.CMAPS[nodekey][2](), vmax=castpar.CMAPS[nodekey][2]()), cmap=cmap), cax=axsplot[0,np.minimum(len(subgraphs),maxnum)])
           
            for k in range(len(addgraphs)):
                if addgraphs[k] in PLOTMODELPROPS.keys():
                    PLOTMODELPROPS[addgraphs[k]](castpar, axsgraph[k])
                elif addgraphs[k] in PLOTNETWPROPS.keys():
                    PLOTNETWPROPS[addgraphs[k]](g, axsgraph[k], castpar.dim//2)
                else:
                    raise ValueError("Incorrect key")
            dat = castpar.getdata()
            tit = "System snapshots at T = "+tformat+". After "+str(rot.snaps["sweepnum"][i])+" sweeps."
            fig.suptitle(title+" - "+tit+" |M| = "+str(round(dat["magn"],2))+" ; n_conn = "+str(round(dat["numconn"],2)))
           
            print(title+" - "+tit[20:],fixslash(savedir)+tformat,loc)
            plotout(fig,title+" - "+tit[20:], fixslash(savedir)+tformat, loc)


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False): #(taken from stackexchange)
    split=True
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]


    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

    
# plot the dynamical evolution of a general property of the rotor
def plotdynamics(dyn,ykey,title="", savedir=None, loc="", writeplot=[True,False], sweeplims = [0,None], ftrans=False):
     
    calc = {(ykey if ykey !="heatcap" else "en"): [], "numconn": [], "temp": dyn[2]}
    
    for tind in range(len(dyn[0])):
        if ykey=="corr":
            if sweeplims[1] == None:
                sweeplims[1] = len(dyn[0][tind]["en"])
    
            calc["corr"].append({"mean": np.sum(corrs(dyn[0][tind],tind,100,sweeplims)["mean"]**2)/10000, "std":0})
            continue
            
            
        d = np.array(dyn[0][tind][ykey])
        
        if sweeplims[1] == None:
            sweeplims[1] = len(d)
        d = d[sweeplims[0]:sweeplims[1]]
        
        calc[ykey].append({"mean": np.mean(d), "std": np.std(d)})
        
        x=np.arange(0,np.shape(d)[0],1)
        
        if ftrans:
            d=np.abs(np.fft.rfft(d)[2:])
            a, b = hl_envelopes_idx(d,2,2)
            d=np.log(d[b])
            x = np.log(np.linspace(0,.5,num=np.shape(d)[0]))
        
        if dyn[1]:
            x+=np.shape(d)[0]    
        
        if ykey != "numconn":
            e = np.array(dyn[0][tind]["numconn"])
            calc["numconn"].append({"mean": np.mean(e[sweeplims[0]:sweeplims[1]]), "std": np.std(e[sweeplims[0]:sweeplims[1]])})      
    
        fig, ax = plt.subplots()
        ax.plot(x,d, label="T="+str(round(dyn[2][tind],2)),color='black')
        ax.set_xlabel("Amount of sweeps")
        ax.set_ylabel(LABS[ykey])
        tit = title+" (T="+str(round(dyn[2][tind],5))+")"
        ax.set_title(tit)
    
        plotout(fig,ykey+" - "+tit, savedir, loc)
        if writeplot[0]:
            plotout(fig, ykey+" - "+tit, savedir,loc)
        if writeplot[1]:
            saverawdata(ax, savedir, loc, fname=ykey+" - "+tit+".txt")
    
    if not ftrans:
        calc[ykey] = calc.pop(list(calc.keys())[0])
        plotprop([calc,sweeplims[1]-sweeplims[0],-1], ykey, title=title, writeplot=writeplot,loc=loc,savedir=savedir)