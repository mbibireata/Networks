# Networks

Simulation of network of neuron networks.

Graph Theoretic routines written by Devin.

Simulation dynamics functions written by Valentin Slepukhin

Phase space exploration by Mihai.

The most up-to-date and readable version of the project is in the file Rhythmogenesis.ipynb

## Instructions for running simulations

1. Initialize Network Connectivity

```
n = 100                         # number of neurons in network
p = 3/4                         # Erdos-Renyi connectivity probability parameter
M = getRandomConnectivity(n, p) # Initialize adjacency matrix
l = matrixOfEdges(M, n)         # Initialize specially formatted adjacency list as matrix
```

2. Initialize Network Parameters 

```
N = 10000
dt = 0.01
T = np.array([i*dt for i in range(N)])

tauv  = 0.01  # Voltage increment time constant  
tauc  = 0.5   # Calcium increment time constant
Vstar = 15    # Inflection point of Voltage Sigmoid (V)
Cstar = 20    # Inflection point of Calcium Sigmoid (Ca conc. arb. units)
V0    = 18    # Initial Voltage parameter (V)
C0    = 18    # Initial Calcium concentration parameter (Ca conc. arb. units)
delV  = 5.    # Voltage Increment (V)
delC  = 0.035 # Calcium concentration increment (Ca conc. arb. units)
r_1   = 70    # Maximal firing rate (Hz)
r_0   = 5     # Basal firing rate (Hz)
g_v   = 5     # Voltage sigmoid steepness parameter (V)
g_c   = 5     # Calcium concentration sigmoid steepness parameter (Ca conc. arb. units)
```

3. Save parameters in np.array or other structure to pass into functions

```
params = np.zeros(15, dtype=float)

params[0]  = N     # Must cast to int later 
params[1]  = n     # Must cast to int later
params[2]  = dt
params[3]  = tauv
params[4]  = tauc
params[5]  = Vstar
params[6]  = Cstar
params[7]  = delV
params[8]  = delC
params[9]  = r_1
params[10] = V0
params[11] = C0
params[12] = g_v
params[13] = g_c
params[14] = r_0
```

4. Initialize network object

```
network = Network(M,l,params)
```

5. Run code to generate phase diagram
    - (neuron activation function keywords "sigmoid" and "step" are accepted)
    - (network connectivity keywords "heterogeneous" and "mean field" are accepted)
    - (x parameter keywords "delV", "delC", "g_v", "g_c" are accepted)
    - (y parameter keyword "n" is accepted)

```
phase = network.phase_diagram("delV", n, delVmin, delVmax, dV, nmin, nmax, dn
                              "full", "heterogeneous", "sigmoid")
```

6. Plot phase diagram

```
# full space plot
cmap = plt.cm.viridis
data = np.flip(phase, axis=0)
norm = plt.Normalize(vmin = data.min(), vmax = data.max())
img = cmap(norm(data))
imshow(img, aspect='auto', extent=[delVmin, delVmax, nmin, nmax])
plt.title("delC = {:.3f}, p = {:.3f}".format(delC, p))
xlabel(r"$\Delta V$")
ylabel(r"Neuron Count, $N$")

# boundary trace plot
for i in phase:
    plt.plot(phase[i][1], phase[i][0])
xlabel(r"$\Delta V$")
ylabel("Number of Neurons, $n$")
```

## Instructions for plotting network connectivity diagrams
1. Install and import networkx library

```
import networkx as nx
```

2. Run a dynamics simulation for a given set of parameters

```
avC, avV, C, V, sump = network.simdyn(network.M, network.l, network.params,
                                      connectivity="heterogeneous, activation="sigmoid")
```

3. Construct directed graph object

```
t = -1                                         # time step from dynamics simulation being plotted

G = nx.DiGraph()
for i in range(network.n): 
    if  V[t, i] > Vstar: # t = -1
        G.add_node(i, color='r', firing=True)  # plot firing neurons in red
    else:
        G.add_node(i, color='y', firing=False) # plot nonfiring neurons in yellow
        
for i in range(network.n): 
    for j in range(network.n): 
        if network.M[i,j] == 1: 
            G.add_edge(i,j)

clist = np.zeros(network.n, dtype=str)
for i in range(network.n):
    color = G.nodes.data()[i]['color']
    clist[i] = color
```

4. Plot network 

```
nx.draw_circular(G, node_size=240, width=0.7, with_labels=True, node_color=clist)
```
