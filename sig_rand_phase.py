#code concists of cells. To run cell, press play button above or Shift+Enter (on Windows)

#This cell is just importing some libraries
import numpy as np
from matplotlib import pylab, mlab, pyplot

from IPython.core.pylabtools import figsize, getfigs

from pylab import *
from numpy import *

style.use('ggplot')
rcParams['figure.figsize'] = (25,13)
from numba import jit

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

#This part is a set of definitions. You can skip it and then come back to see what is the function which I use in the main code

import copy

#Here we construct a random connectivity matrix (single-directional map)
def getRandomConnectivity(N, pct_connected):
    # There is no self coupling so the diagonal must be zero. 
    # The graph is non directed so M must be symmetric
    M = np.random.rand(N**2).reshape(N,N)
    for i in range(N):
        for j in range(N):
            if (i == j):
                M[i,j] = 0
            else:
                if (M[i,j] < pct_connected):
                    M[i,j] = 1
                else:
                    M[i,j] = 0
    return M

# alternate version
def getRandomConnectivity2(N, pct_connected):
    # There are N*(N-1)/2 upper diagonal elements
    rand_elems = np.where(np.random.rand(N*(N-1)/2) < pct_connected, 1, 0)
    M = np.zeros((N,N))
    M[np.triu_indices(N,1)] = rand_elems
    return M + M.T

# Original version with bidirectional mapping 
def getRandomConnectivity3(N, pct_connected):
    # There is no self coupling so the diagonal must be zero. 
    # The graph is non directed so M must be symmetric
    M = np.random.rand(N**2).reshape(N,N)
    for i in range(N):
        for j in range(i, N):
            if (i == j):
                M[i,j] = 0
            else:
                if (M[i,j] < pct_connected):
                    M[i,j] = 1
                else:
                    M[i,j] = 0
                M[j,i] = M[i,j]
    return M


# algorithm for finding k cores
# The basic idea is to recursively remove nodes of connectivity < k until we either have nothing to remove or no nodes left

def hasKcore(M, k):
    # we'll be destroying the array so make a copy to work with
    X = M.copy()
    while(True):
        cur_num_nodes = X.shape[0]
        s = np.sum(X, 0)
        nodes_to_delete = np.where(s < k)[0]
        if (len(nodes_to_delete) == cur_num_nodes):
            # nothing has at least k connections
            
            X = np.delete(X, nodes_to_delete, axis=0)
            X = np.delete(X, nodes_to_delete, axis=1)
            
            return False
            break
        elif (len(nodes_to_delete) == 0):
            # They all have at least k connections, we've found a kcore
            return True
            break
        else:
            X = np.delete(X, nodes_to_delete, axis=0)
            X = np.delete(X, nodes_to_delete, axis=1)

            
# TODO get Kcore indices 

def largestKcore(M):
    # we can begin the search at the least connected node
    connectivity = np.sum(M,0)
    min_kcore = np.min(connectivity)
    max_kcore = np.max(connectivity)
    if max_kcore == 0:
        return 0
    k = min_kcore
    while (hasKcore(M,k)):
        k = k+1
        
    return k - 1
        
    
    
def largestkcore(M):
    # we can begin the search at the least connected node
    connectivity = np.sum(M,0)
    min_kcore = np.min(connectivity)
    max_kcore = np.max(connectivity)
    if max_kcore == 0:
        return 0
    k = min_kcore
    s=1
    while (s==1):
        
        X = M.copy()
        while(True):
            cur_num_nodes = X.shape[0]
            s = np.sum(X, 0)
            nodes_to_delete = np.where(s < k)[0]
            if (len(nodes_to_delete) == cur_num_nodes):
            # nothing has at least k connections
            
                X = np.delete(X, nodes_to_delete, axis=0)
                X = np.delete(X, nodes_to_delete, axis=1)
            
                s=0
                break
            elif (len(nodes_to_delete) == 0):
            # They all have at least k connections, we've found a kcore
                s=1
                break
            else:
                X = np.delete(X, nodes_to_delete, axis=0)
                X = np.delete(X, nodes_to_delete, axis=1)
        k = k+1
        
    return k - 1


def kcoreitself(M, k):
    # we'll be destroying the array so make a copy to work with
    X = M.copy()
    
    while(True):
        cur_num_nodes = X.shape[0]
        s = np.sum(X, 0)
        nodes_to_delete = np.where(s < k)[0]
        if (len(nodes_to_delete) == cur_num_nodes):
            
            X = np.delete(X, nodes_to_delete, axis=0)
            X = np.delete(X, nodes_to_delete, axis=1)
            # nothing has at least k connections
            return X
            break
        elif (len(nodes_to_delete) == 0):
            # They all have at least k connections, we've found a kcore
            return X
            break
        else:
            X = np.delete(X, nodes_to_delete, axis=0)
            X = np.delete(X, nodes_to_delete, axis=1)



#here we get the largest connected component of  the graph

def largestcomponent(l,N):
    n=np.random.randint(0,N)
    
    boundary=[n]
    b=1
    comp=[n]
    c=1
   
    while(b>0):
        a=boundary.pop()
        
        oldvertex=int(a)
        
        deg=l[oldvertex,0]
        degree=int(deg)
        #print(degree)
        for j in range(degree):
            newvertex=l[oldvertex,j+1]
            q=comp.count(newvertex)
            #print(newvertex)
            q=int(q)
            #print(q)
            if (q==0):
                comp.append(newvertex)
                boundary.append(newvertex)
                #print(newvertex)
                #print(boundary)
            
        b=len(boundary)
            
    #sizeofcomp=len(comp)
    return(comp)

def matrixOfEdges(M,N):
    N = int(N)
    E=np.zeros(N**2).reshape(N,N)  #prepare array with zeros
    k=np.sum(M,0) #array with degree of each vertex
    for i in range(N):
        a=int(k[i]) #degree of the current vertex
        E[i,0]=a  #we put it to the zero row of matrix of edges
    for i in range(N):
        a=int(k[i])
        q=1
        for j in range(N):
            if (M[i,j]==1):
                E[i,q]=j #all the next elements in current column are number of vertices current vertex is connected to
                q=q+1
    return(E)

def simdyn(M,l,params):
    return simDynamics(M,l,params)

@jit(nopython=True)
def simDynamics(M,l,params):
    N=int(params[0])
    n=int(params[1])
    dt=params[2]
    tauv=params[3]
    tauc=params[4]
    Vstar=params[5]
    Cstar=params[6]
    delV=params[7]
    delC=params[8]
    r=params[9]
    V0=params[10]
    C0=params[11]
    V=zeros(N*n).reshape(N,n)  #now we have potencial and calcium concentration for each neuron
    C=zeros(N*n).reshape(N,n)
    P=zeros(n)  # function P(V), sigmoid
    sump=zeros(n)  #sum of P over the vertices
    avV=zeros(N) #average potencial
    avC=zeros(N)  #average calcium
    #res = int(params[12]) #resulting classification 
    
    for i in range(n):
        V[0,i]=V0 #initial conditions
        C[0,i]=C0 
        avV[0]=n*V0  #we will divide to n later
        avC[0]=n*C0
    for i in range (N-1):
        for j in range(n):
            P[j] = 1/(1+exp((Vstar-V[i,j])*0.2))
        for j in range(n):
            a=int(l[j,0])
            for k in range(a):
                b=int(l[j,k])
                sump[j]=sump[j]+P[b]  #if neuron connected to j-th neuron it participates in sum
        for j in range(n):
            V[i+1,j]=(-V[i,j]/tauv + delV*r*sump[j]*(1/(1+exp(C[i,j]-Cstar))) )*dt + V[i,j]
            C[i+1,j]=(-C[i,j]/tauc + delC*r*sump[j])*dt + C[i,j]
            avC[i+1]=avC[i+1]+C[i+1,j]
            avV[i+1]=avV[i+1]+V[i+1,j]
        for j in range(n):
            sump[j]=0
    for i in range (N):
        avC[i]=avC[i]/n
        avV[i]=avV[i]/n
        
        #plot([avC[i],avC[i+1]],[avV[i],avV[i+1]])    
    return(avC,avV)

#TODO: Fix Classification function
###Classification 1 : Simple###
def classify(V, params):
    #Set up some globals 
    
    N = int(params[0])
    n = int(params[1])
    
    #Network uniformly connected but vectorized lets us scale later 
    #for i in range (N/2, N):
    max_V = max(V[N/2:N])
    min_V = min(V[N/2:N])
   
            
    V_delta = max_V - min_V
    
    if V_delta < 1:
        if V[N-1] < 15:
            #params[12] = 2
            return 2
        else:
            #params[12] = 3
            return 3
    else:
        #params[12] = 1
        return 1

#Removes one neuron from the network (cull an x,y axis). 
def killNeuron(N,M_prime):
    M = copy.deepcopy(M_prime)
    #Randomly pick a neuron to remove
    #x = np.random.randint(N)
    x = 1
    M = np.delete(M,x,axis=0)
    M = np.delete(M,x,axis=1)
    return M

#TODO: Kill N neurons from the network systematically instead of one at a time. 
def killNeuron2(to_cull, M_prime):
    M = copy.deepcopy(M_prime) 
    #Systemically cull certain neuron from the matrix
    x = 0    
    for i in range (to_cull):
        M = np.delete(M,x,axis=0)
        M = np.delete(M,x,axis=1)
    return M

#actual run for random matrix

N=100000
n=int(100) #100
dt=0.001
T=zeros(N)
for i in range(N):
    T[i]=i*dt
p = .75 #Connectivity rate

M=getRandomConnectivity(n,p) #args(n,p)
l=matrixOfEdges(M,n)
M_prime = copy.deepcopy(M)
l_prime = copy.deepcopy(l)
#curr_n = n #Current number of neurons in network 
to_cull = 0

tauv=0.01 #0.01
tauc=0.5 #0.5
Vstar=15 #Constant (according to experiments) 15
Cstar=15 #15
V0=18 #18
C0=18 #18

delV= 3.0 #1.2
delC=0.015 #0.015
r=70 #70

params=zeros(13)
params[0]=N
params[1]=n
params[2]=dt
params[3]=tauv
params[4]=tauc
params[5]=Vstar
params[6]=Cstar
params[7]=delV
params[8]=delC
params[9]=r
params[10]=V0
params[11]=C0
params[12]=p #connenctivity rate from above. 


phasen = []
phaseV = []
kCore = []

dV = 0.01
delVmax = 5.0

while True:
    C,V = simdyn(M_prime,l_prime,params)
    check = classify(V,params)
    while True:
        #import pdb; pdb.set_trace()
        params[1] -= 1
        #curr_n = curr_n-1
        to_cull += 1
        M_prime = killNeuron2(to_cull, M)
        l_prime = matrixOfEdges(M_prime, params[1])
        if params[1] <= 1:
            params[7] += dV
            params[1] = n
            to_cull = 0
            M_prime = copy.deepcopy(M)
            l_prime = copy.deepcopy(l)
            #curr_n = n
            break
        C,V = simdyn(M_prime,l_prime,params)
        check2 = classify(V,params)
        if check2 > check:
            print(params[7])
            phasen.append(params[1])
            phaseV.append(params[7])
            kCore.append(largestKcore(M_prime))
            params[7] += dV
            if params[1] < n-3:
                to_cull -= 3
                params[1] += 3 #Changed from params[1] = n to track boundary 
                M_prime = killNeuron2(to_cull, M)

            else:
                to_cull = 0
                params[1] = n 
                M_prime = copy.deepcopy(M)
                
            l_prime = copy.deepcopy(l)
            break
            #TODO: Fix to go back to matrix that we had 3 neurons ago. 
            #M_prime = killNeuron2(to_cull, M) 
            #l_prime = copy.deepcopy(l)
            #curr_n = n
            #break

    if params[7] > delVmax:
        break

#TODO: HA to SO phase.... find the k that it needs to bump up
print('loop complete')
if len(phasen) > 0 and len(phaseV) > 0:
    figure(1)
    ylabel('n, number of neurons')
    xlabel('delV')
    plot(phaseV,phasen, '-o')
    savefig('sig_rand_p_016.png')

    figure(2)
    ylabel('largest k core') 
    xlabel('delV') 
    plot(phaseV, kCore, '-0') 
    savefig('sig_rand_p_016_kcore.png')
else:
    print('no transition')
