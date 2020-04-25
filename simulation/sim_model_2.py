# This is the simulation of our evolving RS model under the SECOND framework of our assumptions on edge weights.
import numpy as np
import random
import matplotlib.pyplot as plt
import powerlaw
import pandas as pd

class assumption_2nd:
    # initializing the whole model
    def __init__(self, beta, iterations, rating_scale, Cu, Ci, Unum, Inum, K, L, C):
        self.init_paramter(beta, iterations, rating_scale, Cu, Ci, Unum, Inum, K, L, C)
        self.init_assumption()
        k = self.stat()
        self.iterate()
        res = self.stat()
        tdu = self.calcdegree_user()
        twu = self.calcweight_user()
        tdi = self.calcdegree_item()
        twi = self.calcweight_item()
        
        k = (res, tdu, twu, tdi, twi)
        x = np.zeros(self.rating_scale)
        self.degseq_user = np.zeros(self.iterations + 1)
        self.weiseq_user = np.zeros(self.iterations + 1)
        self.degseq_item = np.zeros(self.iterations + 1)
        self.weiseq_item = np.zeros(self.iterations + 1)
        
        x[:res.size] = x[:res.size] + res
        self.degseq_user[:min(self.iterations+1,k[1].size)] = self.degseq_user[:min(self.iterations+1,k[1].size)] + k[1][:min(self.iterations+1,k[1].size)]
        self.weiseq_user[:min(self.iterations+1,k[2].size)] = self.weiseq_user[:min(self.iterations+1,k[2].size)] + k[2][:min(self.iterations+1,k[2].size)]
        self.degseq_item[:min(self.iterations+1,k[3].size)] = self.degseq_item[:min(self.iterations+1,k[3].size)] + k[3][:min(self.iterations+1,k[3].size)]
        self.weiseq_item[:min(self.iterations+1,k[4].size)] = self.weiseq_item[:min(self.iterations+1,k[4].size)] + k[4][:min(self.iterations+1,k[4].size)]
        np.set_printoptions(threshold=np.inf)
        xind = np.zeros(self.iterations + 1)
        for i in range(1,self.iterations + 1):
            xind[i] = xind[i-1] + 1
        self.xind_user = xind
        self.xind_item = xind
        print("finish all the staff")
        
    #Initial settings of parameters in our weighted bipartite graph model B(U,I).
    def init_paramter(self, beta, iterations, rating_scale, Cu, Ci, Unum, Inum, K, L, C):
        #Initial settings of parameters in our weighted bipartite graph model B(U,I).
        self.beta = beta # the probability to add a new vertex in U
        self.iterations = iterations # the number of iterations to run the simulation
        self.rating_scale = rating_scale # the preassigned rating scale
        self.Cu = Cu # the least number of edges connected to vertices in U
        self.Ci = Ci # the least number of edges connected to vertices in I
        self.Unum = Unum # the number of vertices in U in the initial graph at t=0
        self.Inum = Inum # the number of vertices in I in the initial graph at t=1
        self.K = K # the number of basic user type in our assumption
        self.L = L # the number of basic item level in our assumption
        
        self.C = C # the number of adding edge
        
        self.Hui = np.zeros((rating_scale,K,L)) # the rating pmf for the pair of K user types and L item levels
        self.Fmean = np.zeros((K,)) # the mean of the distribution of users' weight vector (assumed to be Gaussian)
        self.Gmean = np.zeros((L,)) # the mean of the distribution of items' weight vector (assumed to be Gaussian)
        self.edges = np.zeros((iterations+50,iterations+50), dtype=int) # the matrix storing edge information
        self.Uweight = np.zeros((iterations+50,K)) # the matrix storing users' weight vectors
        self.Iweight = np.zeros((iterations+50,L)) # the matrix storing items' weight vectors
    
    # Initalization of the sampling of edge weights from the mixture distribution
    def init_weightgenerator(self):
        # include K,L,Huser,Hitem,Hui,rating_scale,Fmean,Gmean
        self.Hui = np.random.sample((self.rating_scale, self.K, self.L))
        Huisubsum = np.sum(self.Hui,axis=0)
        Huisubsum = np.array([Huisubsum] * self.rating_scale)
        self.Hui = self.Hui/Huisubsum
        
        self.Huser = np.random.sample((self.K, self.rating_scale))
        Husersubsum = np.sum(self.Huser, axis=1)
        Husersubsum = np.array([Husersubsum] * self.rating_scale)
        Husersubsum = np.transpose(Husersubsum)
        self.Huser = self.Huser/Husersubsum
        
        self.Hitem = np.random.sample((self.L, self.rating_scale))
        Hitemsubsum = np.sum(self.Hitem, axis=1)
        Hitemsubsum = np.array([Hitemsubsum] * self.rating_scale)
        Hitemsubsum = np.transpose(Hitemsubsum)
        self.Hitem = self.Hitem/Hitemsubsum
        
        self.Fmean = np.random.sample(self.K,)
        self.Fmean = self.Fmean/np.sum(self.Fmean)
        self.Gmean = np.random.sample(self.L,)
        self.Gmean = self.Gmean/np.sum(self.Gmean)
    
    #Sample edge weight for the edge between vertices (Uid, Iid)
    def weightgenerator(self, Uid, Iid):
        # include K,L,Huser,Hitem,Hui,rating_scale,Fmean,Gmean
        Uw = self.Uweight[Uid:Uid+1,:]
        Iw = self.Iweight[Iid:Iid+1,:]
        Uw = np.transpose(Uw)
        Hr = np.dot(Uw,Iw)
        Hr = np.array([Hr] * self.rating_scale)
        Hr = self.Hui * Hr
        Hr = np.sum(Hr,axis=1)
        Hr = np.sum(Hr,axis=1)
        R = np.random.choice(self.rating_scale, 1, p=Hr) + 1
        return R
    
    # Initialization for the inital simple graph at t=0
    def init_assumption(self):
        # include edges,Unum,Inum,Uweight,Iweight,K,L,Fmean,Gmean
        print("Initializing...", end="")
        self.init_weightgenerator()
        for i in range(self.Unum):
            Utmp = np.random.normal(self.Fmean, 0.1)
            Utmp[Utmp<0]=0
            Utmp = Utmp/np.sum(Utmp)
            self.Uweight[i,:]=Utmp
        for i in range(self.Inum):
            Itmp = np.random.normal(self.Gmean, 0.1)
            Itmp[Itmp<0]=0
            Itmp = Itmp/np.sum(Itmp)
            self.Iweight[i,:]=Itmp
        self.edges = np.zeros((self.iterations+50, self.iterations+50), dtype=int)
        # We can assume that axis=1 is user sequence and the axis=0 is the item sequence
        for i in range(self.Unum):
            for j in range(self.Inum):
                self.edges[i,j] = self.weightgenerator(i,j)
        print("Done.")
        
    # Select "prototype" from the existing vertex group
    def prototype(self, arr, nb):
        return np.count_nonzero(arr.cumsum() < nb)

    # Conduct Edge-copy and assign new edge weights
    def copyedge(self, template, desired, p_prime):
        ls = []
        new2old = template.nonzero()[0]
        tmp = template[new2old].astype(float)
        for i in range(desired):
            tmp /= tmp.sum()
            sampled = np.nonzero(np.random.multinomial(1, tmp))[0][0]
            ls.append(sampled)
            tmp[sampled] = 0
        ls.sort()
        return new2old[ls]

    # Add new vertices to U (respectively. I)
    def addnode(self, nb_axis):
        # include edges,Unum,Inum,Fmean,Gmean
        weightsum = np.sum(self.edges[:self.Unum,:self.Inum], axis=nb_axis)
        totalsum = np.sum(weightsum)
        randnum = np.random.randint(1, totalsum+1)
        p_prime = self.prototype(weightsum, randnum)
        weighted = np.zeros(1)
        if nb_axis == 1:
            Utmp = np.random.normal(self.Fmean, 0.1)
            Utmp[Utmp<0] = 0
            Utmp = Utmp/np.sum(Utmp)
            self.Uweight[self.Unum,:]=Utmp
            template = self.edges[p_prime,:self.Inum]
            desired = self.Cu
            idx = self.copyedge(template, desired, p_prime)
            new = np.zeros(template.shape[0], dtype=int)
            for i in range(idx.shape[0]):
                new[idx[i]]= self.weightgenerator(self.Unum, idx[i])
            self.edges[self.Unum,:self.Inum] = new
            self.Unum = self.Unum + 1
        else:
            Itmp = np.random.normal(self.Gmean, 0.1)
            Itmp[Itmp<0] = 0
            Itmp = Itmp/np.sum(Itmp)
            self.Iweight[self.Inum,:]=Itmp
            template = self.edges[:self.Unum, p_prime]
            desired = self.Ci
            idx = self.copyedge(template, desired, p_prime)
            new = np.zeros(template.shape[0], dtype=int)
            for i in range(idx.shape[0]):
                new[idx[i]]= self.weightgenerator(idx[i], self.Inum)
            self.edges[:self.Unum,self.Inum] = new
            self.Inum = self.Inum + 1
        
    # Add new edges to Graph
    def addedge(self):
        # include edges,Unum,Inum
        randnum_user = random.randint(1,self.Unum-1)
        randnum_item = random.randint(1,self.Inum-1)
        self.edges[randnum_user,randnum_item] = random.randint(1, self.rating_scale)
        
    # Evolution of U (or I)
    def evolution(self):
        randnum = np.random.rand()
        if randnum < self.beta:
            self.addnode(1)
        else:
            self.addnode(0)
        for i in range(self.C):
            self.addedge()
            # pass

    # Iterate 
    def iterate(self):
        print("Begin iteration...", end="")
        for i in range(self.iterations):
            self.evolution()
        print("Done")

    # Gather statistic information
    def stat(self):
        # include edges
        tmps = self.edges.flatten().astype(int)
        count = np.bincount(tmps)
        count = count[1:]
        count = 1.0*count/count.sum()
        return count

    # Calculate user degree distributions
    def calcdegree_user(self):
        # include edges
        sumdegree = self.edges.astype(bool).sum(axis=1)
        return np.bincount(sumdegree)

    # Calculate user weight distributions
    def calcweight_user(self):
        # include edges
        sumdegree = self.edges.sum(axis=1)
        return np.bincount(sumdegree)

    # Calculate item degree distributions
    def calcdegree_item(self):
        # include edges
        sumdegree = self.edges.astype(bool).sum(axis=0)
        return np.bincount(sumdegree)

    # Calculate item weight distributions
    def calcweight_item(self):
        # include edges
        sumdegree = self.edges.sum(axis=0)
        return np.bincount(sumdegree)
    
    def get_distribution(self, target="user"):
        if target == "item":
            return self.degseq_item, self.weiseq_item, self.xind_item
        else:
            return self.degseq_user, self.weiseq_user, self.xind_user
    
    def get_graph(self):
        return self.edges, self.Inum, self.Unum
    
    
def get_pvalue_alpha_xmin_2(seq):
    results = powerlaw.Fit(seq)
    alpha = results.power_law.alpha
    xmin = results.power_law.xmin
    R, p_value = results.distribution_compare('power_law', 'lognormal')
    print("p_value:", p_value, "alpha:", alpha, "xmin:", xmin)
    return p_value, alpha, xmin