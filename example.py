import pdb
import numpy as np
import matplotlib.pyplot as plt
from process import VectorAutoregressiveProcess

# INSTANTIATION
num_series = 2
num_lags = 3
seed = 0

# PROCESS
process = VectorAutoregressiveProcess(num_series,num_lags,seed=seed)

## manual setting
process.regressors = lambda x: np.array([0,0])
process.covariance = np.diag([0.2,0.1])
process.mean = np.array([0.01,0.002])
process.model_parameters['A2'] = np.array([[0.2,0.1],[0.6,0.1]])
process.stabilize(verbose=True,vis=True)
### To turn off the visualization of the stabilization you can sef the variables vis | verbose to False or simply call: process.stabilize()

## var generator
var = process.generator(num_realizations=100)

### auxiliar variables
x = []
s1 = []
s2 = []

## reading the genertor through next
### this is useful for online learning settings

while True:
    try:
        n,y = next(var)
    except StopIteration:
        break

    x.append(n)
    s1.append(y[0])
    s2.append(y[1])

## reading the generator through for loop (uncomment if you want to use it)
### this is useful to load all the data in advance

# for n,y in var:
#     x.append(n)
#     s1.append(y[0])
#     s2.append(y[1])


# PLOTs
fig, (ax1,ax2) = plt.subplots(2)
fig.suptitle('Synthetic VAR({}) data with {} series'.format(num_lags,num_series))
ax1.plot(x,s1,'xb')
ax1.grid()
ax2.plot(x,s2,'xk')
ax2.grid()
plt.show()