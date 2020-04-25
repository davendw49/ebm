# This is the simulation of our evolving RS model under the FIRST framework of our assumptions on edge weights.

import math
import numpy as np
import matplotlib.pyplot as plt

#Initial settings of parameters in our weighted bipartite graph model B(U,I).
beta = 0.6 # the probability to add a new vertex in U
iterations = 3000 # the number of iterations to run the simulation
rating_scale = 5 # the preassigned rating scale
Cu = 10 # the least number of edges connected to vertices in U
Ci = 20 # the least number of edges connected to vertices in I
Unum = 20 # the number of vertices in U in the initial graph at t=0
Inum = 10 # the number of vertices in I in the initial graph at t=1
K = 10 # the number of basic user type in our assumption
L = 5 # the number of basic item level in our assumption
Huser = np.zeros((K,rating_scale)) # the rating pmf for the K user types
Hitem = np.zeros((L,rating_scale)) # the rating pmf for the L item levels
Fmean = np.zeros((K,)) # the mean of the distribution of users' weight vector (assumed to be Gaussian)
Gmean = np.zeros((L,)) # the mean of the distribution of items' weight vector (assumed to be Gaussian)
edges = np.zeros((iterations+50,iterations+50), dtype=int) # the matrix storing edge information

# Initalization of the sampling of edge weights from the mixture distribution
def init_weightgenerator():
    global K,L,Huser,Hitem,rating_scale,Fmean,Gmean
    Huser = np.random.sample((K,rating_scale))
    Husersubsum = np.sum(Huser,axis=1)
    Husersubsum = np.array([Husersubsum]*rating_scale)
    Husersubsum = np.transpose(Husersubsum)
    Huser = Huser/Husersubsum
    Hitem = np.random.sample((L,rating_scale))
    Hitemsubsum = np.sum(Hitem,axis=1)
    Hitemsubsum = np.array([Hitemsubsum]*rating_scale)
    Hitemsubsum = np.transpose(Hitemsubsum)
    Hitem = Hitem/Hitemsubsum
    Fmean = np.random.sample(K,)
    Fmean = Fmean/np.sum(Fmean)
    Gmean = np.random.sample(L,)
    Gmean = Gmean/np.sum(Gmean)

# Sample edge weight(s) for new users in U from the mixture distribution
def userweightgenerator(nb):
    global K,L,Huser,Hitem,rating_scale,Fmean,Gmean
    Uvec = np.random.normal(Fmean,0.1)
    Uvec[Uvec<0]=0
    Uvec = Uvec/np.sum(Uvec)
    Uvec = np.array([Uvec]*rating_scale)
    Uvec = np.transpose(Uvec)
    Hu = Huser*Uvec
    Hu = np.sum(Hu,axis=0)
    R = np.random.choice(rating_scale,nb,p=Hu)+1
    return R

# Sample edge weight(s) for new items in I from the mixture distribution
def itemweightgenerator(nb):
    global K,L,Huser,Hitem,rating_scale,Fmean,Gmean
    Ivec = np.random.normal(Gmean,0.1)
    Ivec[Ivec<0]=0
    Ivec = Ivec/np.sum(Ivec)
    Ivec = np.array([Ivec]*rating_scale)
    Ivec = np.transpose(Ivec)
    Hi = Hitem*Ivec
    Hi = np.sum(Hi,axis=0)
    R = np.random.choice(rating_scale,nb,p=Hi)+1
    return R

# Initialization for the inital simple graph at t=0
def init():
    print ('Initalizing...')
    global edges,Unum,Inum
    init_weightgenerator()
    edges = np.zeros((iterations+50,iterations+50), dtype=int)
    for i in range(Unum):
        edges[i,0:Inum] = userweightgenerator(Inum)
    print ('Done.')
    
# Select "prototype" from the existing vertex group
def prototype(arr, nb):
    return np.count_nonzero(arr.cumsum() < nb)

# Conduct Edge-copy and assign new edge weights
def copyedge(template, desired,p_prime):
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
def addnode(nb_axis):
    global edges,Unum,Inum
    weightsum = np.sum(edges[:Unum,:Inum], axis=nb_axis)
    totalsum = np.sum(weightsum)
    randnum = np.random.randint(1, totalsum+1)
    p_prime = prototype(weightsum, randnum)
    weighted = np.zeros(1)
    if nb_axis == 1:
        template = edges[p_prime, :Inum]
        desired = Cu
        weighted = userweightgenerator(template.shape[0])
    else:
        template = edges[:Unum, p_prime]
        desired = Ci
        weighted = itemweightgenerator(template.shape[0])
    idx = copyedge(template, desired,p_prime)
    new = np.zeros(template.shape[0],dtype=int)
    new[idx] = weighted[idx]
    if nb_axis == 1:
        edges[Unum,:Inum] = new
        Unum = Unum + 1
        # print(list(new), file=f)
    else:
        edges[:Unum,Inum] = new
        Inum = Inum + 1
        # print(list(new), file=f)


# Evolution of U (or I)
def evolution():
    randnum = np.random.rand()
    if randnum < beta:
        addnode(1)
    else:
        addnode(0)
    
# Iterate 
def iterate():
    print ('Begin iteration...')
    for i in range(iterations):
        evolution()
    print ('Done')
    
# Gather statistic information
def stat():
    global edges
    tmps = edges.flatten().astype(int)
    count = np.bincount(tmps)
    count = count[1:]
    count = 1.0*count/count.sum()
    return count

# Calculate degree distributions
def calcdegree():
    global edges
    sumdegree = edges.astype(bool).sum(axis=1)
    return np.bincount(sumdegree)

# Calculate vertex weight distributions
def calcweight():
    global edges
    sumdegree = edges.sum(axis=1)
    return np.bincount(sumdegree)

# Run the simulation for one time
def once():
    global edges,Unum,Inum
    init()
    k = stat()
    iterate()
    res = stat()
    deg = calcdegree()
    weights = calcweight()
    print('--------------------------')
    print(len(res), len(deg), len(weights))
    # for item in edges:
        # print(len(item))
        # print(item.tolist(), file=f)
    return (res, deg, weights)

def main():
    x = np.zeros(rating_scale)
    deg = np.zeros(iterations+1)
    wei = np.zeros(iterations+1)
    for i in range(1):
        k = once()
        x[:k[0].size] = x[:k[0].size] + k[0]
        deg[:min(iterations+1,k[1].size)] = deg[:min(iterations+1,k[1].size)] + k[1][:min(iterations+1,k[1].size)]
        wei[:min(iterations+1,k[2].size)] = wei[:min(iterations+1,k[2].size)] + k[2][:min(iterations+1,k[2].size)]
    np.set_printoptions(threshold=np.nan)
    print ('Degree sequence:')
    # print (deg)
    print ('================================')
    print ('Vertex weight sequence:')
    # print (wei)
    xind = np.zeros(iterations+1)
    for i in range(1,iterations+1):
        xind[i]=xind[i-1]+1

    for i in range(1, 100):#len(wei[1:])+1
        if wei[i]>0:
            plt.scatter(xind[i],wei[i], c='blue', alpha=0.5)
    plt.title('weight and degree distribution, beta=' + str(beta))
    plt.xlabel('Vertex weight')
    plt.ylabel('Number')
    plt.xscale('log')
    plt.yscale('log')
    # plt.show()

    for i in range(1, 100):#len(deg[1:])+1
        if deg[i]>0:
            plt.scatter(xind[i],deg[i], c='red', alpha=0.5)
    # plt.title('Degree distribution, beta='+str(beta))
    plt.xlabel('Degree')
    plt.ylabel('Number')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

if __name__=="__main__":
    main()
