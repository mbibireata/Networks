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
