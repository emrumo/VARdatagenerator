import pdb
import numpy as np
import matplotlib.pyplot as plt

class VectorAutoregressiveProcess:

    def __init__(self,
                num_series,
                num_lags,
                seed=None):
        
        ## reproducibility
        np.random.seed(seed)

        ## attributes
        self.num_series = num_series
        self.num_lags = num_lags

        ### default initialization
        self.regressors = lambda x: np.array([0]*num_series) 
        self.regressor_coefficients = np.eye(num_series)
        self.mean = np.zeros(num_series)
        self._random_initialization() #### Initializes model_parameters and covariance
        self.set_initial_conditions(np.zeros(num_series))

        ## stabilization
        self.stabilize()
        
    # PUBLIC
    def generator(self,num_realizations=100):
        """ Produces a VAR generator
        Args:
            num_realizations (int, optional): maximum number of realizations (observations) for the VAR generator. Defaults to 100.
        Yields:
            n,y (int,float): [n] is a value within a uniformly spaced grid (by integer units) and [y] constains the VAR realizations
        """

        for n in range(1,num_realizations+1):

            y = np.zeros(self.num_series)

            ## A_1 @ y_{t-1} + ... + A_p @ y_{t-p} 
            for idx in range(self.num_lags):
                y += self.model_parameters['A{}'.format(idx)] @ self.memory[:,idx] 

            ## regressors
            y += self.regressor_coefficients @ self.regressors(n) 

            ## noise
            y += np.random.multivariate_normal(self.mean,self.covariance)

            ## update memory
            self.memory = np.concatenate(( np.expand_dims(y,axis=1) , self.memory ),axis=1)[:,0:-1]

            yield n,y    
    

    def stabilize(self,eps=1e-1,verbose=False,vis=False):
        """ Stabilizes the VAR process / generator
        Args:
            eps (float, optional): Constant for numerical stability. Defaults to 1e-1.
            verbose (bool, optional): Text description. Defaults to False.
            vis (bool, optional): Plots the spectral radius for stability and the eigenvalues of the companion form. Defaults to False.
        """

        param_matrix = np.concatenate(([*self.model_parameters.values()]),axis=1)
        row_norm = np.linalg.norm(param_matrix,axis=1,ord=1)

        if (row_norm < 1).all():
            if verbose:
                print('The process is already stable')
            if vis:
                self._unit_circle()

        else:
            
            if vis:
                self._unit_circle(title='Before stabilization')

            ## normalizing induced inft-norm
            for row,norm in enumerate(row_norm):
                if norm > 1:
                    for key in self.model_parameters:
                        self.model_parameters[key][row,:] = self.model_parameters[key][row,:] / (norm + eps) 

            if verbose:
                print('The process has been stabilized')
            if vis:
                self._unit_circle(title='After stabilization')
        

    def set_initial_conditions(self,initial_conditions):
        """ Sets the initial conditions of the VAR process
        Args:
            initial_conditions (array): Array of dimension [num_series]
        """

        self.memory = np.concatenate(( np.expand_dims(initial_conditions,axis=1) , np.zeros((self.num_series,self.num_lags-1)) ),axis=1)

    # PRIVAT
    def _unit_circle(self,title=''):
        """ Plots the unit circle (for visualizing stability) if specified in the public function [stabilize()]
        Args:
            title (str, optional): Title of the plot to diferentiate between before and after stabilization. Defaults to ''.
        """

        ## figure
        plt.figure(figsize=(6,6))

        plt.title(title)
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.grid()

        ## unit circle
        plt.plot(0,0,'kx')
        t = np.linspace(-2*np.pi,2*np.pi,1000)
        plt.plot(np.cos(t),np.sin(t),'k',label='unit circle')
        
        ## eigenvalues
        G = self._gamma_matrix()
        L = np.linalg.eigvals(G)

        flag = True
        for lamb in L:
            if flag:
                label_name = 'eigenvalue'
                flag = False
            else:
                label_name = '_eigenvalue'

            plt.plot(lamb.real,lamb.imag,'bo',mfc='none',label=label_name)

        plt.legend(loc='upper right')
        plt.show()

    def _random_initialization(self):
        """ Initializes (randomly) the model parameters and the covariance of the VAR process.
        """

        self.model_parameters = {'A{}'.format(n):np.random.random((self.num_series,self.num_series)) for n in range(self.num_lags)}
        self.covariance = np.diag( np.random.random(self.num_series) )

    def _gamma_matrix(self):
        """ Gamma matrix of the companion form for the VAR process
        Returns:
            Gamma matrix (array): Gamma matrix of the companion form. Array of dimension 
        """
          
        upper = np.concatenate(([*self.model_parameters.values()]),axis=1)
        lower = np.concatenate(( np.eye( (self.num_lags-1)*self.num_series ) , np.zeros(( (self.num_lags-1)*self.num_series , self.num_series)) ),axis=1)
        G = np.concatenate((upper, lower), axis=0)

        return G


