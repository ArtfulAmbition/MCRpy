
import os
# Configure TensorFlow/C++ logging and oneDNN before any TensorFlow import.
# - TF_CPP_MIN_LOG_LEVEL: 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
#   Setting to '3' hides INFO and WARNING, keeping only ERROR messages.
# - TF_ENABLE_ONEDNN_OPTS=0 disables oneDNN custom-op informational messages.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.termination import NoTermination
from pymoo.operators.repair.rounding import RoundingRepair
from mcrpy.optimizers.Optimizer import Optimizer
from mcrpy.src.MutableMicrostructure import MutableMicrostructure
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
# Also set Python-side logger to ERROR for tensorflow logger (if TF is imported later)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()

def mpi_logging(message:str='', mode:str='default', end:str='\n'):
    #assert isinstance(message,str)
    if rank==0:
        if mode.lower() == 'debug':
            logging.debug(message)
        elif mode.lower() == 'info':
            logging.info(message)
        elif mode.lower() == 'print':
            print(message,end=end)
        elif mode.lower() == 'default':
            logging.info(message)
            print(message,end=end)
        else:
            raise TypeError(f'mode {mode} not implemented.')

class DiverseRandomSampling(Sampling):
    """Custom sampling to ensure phase diversity in initial population.
    
    Optionally creates an initial connected path for the phase_of_interest
    to provide a better starting point for tortuosity optimization.
    """
    
    def _do(self, problem, n_samples, start_with_connected_path=False, **kwargs):
        """Generate diverse initial population with mixed phases.
        Optionally ensures connectivity for phase_of_interest.
        """
        def calculate_X():
            n_phases = int(problem.xu[0]) + 1
            
            if start_with_connected_path:
                X = np.zeros((n_samples, problem.n_var), dtype=int)
                
                ms_shape = problem.ms_shape
                dimensionality = len(ms_shape)
                phase_of_interest = getattr(problem, 'phase_of_interest', 0)
                direction = getattr(problem, 'direction', 0)
                
                for i in range(n_samples):
                    # Generate random microstructure
                    init_ms = np.random.randint(0, n_phases, ms_shape).astype(int)
                    self._add_connected_path(init_ms, phase_of_interest, direction, dimensionality)
                    
                    X[i] = init_ms.flatten()
                
                return X
            else:
                X = np.random.randint(0,n_phases,size=(n_samples, problem.n_var), dtype=int)
            return X
        
        if comm.rank==0:
            X_to_broadcast = calculate_X()
        else:
            X_to_broadcast = None
        X = comm.bcast(X_to_broadcast, root=0)
        return X
    
    def _add_connected_path(self, ms, phase, direction, dim):
        """Add a connected path for the given phase along the specified direction."""
        
        other_directions = [other_direction for other_direction in (0,1,2) if other_direction!= direction]
        
        if dim == 2:
            other_direction = other_directions[0]
            rand_int = np.random.randint(0, ms.shape[other_direction])
            if direction == 0:  # x-direction: set full row to connect left to right
                ms[rand_int, :] = phase
            elif direction == 1:  # y-direction: set full column to connect top to bottom
                ms[:, rand_int] = phase
        elif dim == 3:
            other_direction1 = other_directions[0]
            other_direction2 = other_directions[1]
            rand_int1 = np.random.randint(0, ms.shape[other_direction1])
            rand_int2 = np.random.randint(0, ms.shape[other_direction2])
            if direction == 0:  # x-direction: set full row in random depth to connect left to right
                ms[rand_int1, rand_int2, :] = phase
            elif direction == 1:  # y-direction: set full column in random depth to connect top to bottom
                ms[rand_int1, :, rand_int2] = phase
            elif direction == 2:  # z-direction: set full depth in random row/col to connect front to back
                ms[:, rand_int1, rand_int2] = phase

class PhaseBitflip(Mutation):
    """Safe bit-flip style mutation for integer phase variables.

    Instead of flipping binary bits (which may produce negative values
    when interpreted), this mutation replaces selected variables with a
    random phase in the valid range [0, n_phases-1].
    """
    def __init__(self, prob=0.5, prob_var=0.1, n_phases: int = 2):
        super().__init__()
        self.prob = prob
        self.prob_var = prob_var
        self.n_phases = int(n_phases)

    def _do(self, problem, X, **kwargs):
        # X: (n_individuals, n_var)
        if self.n_phases <= 1: #if there is just one phase (only zeros), don't do anything. Doesn't make a lot of sense, just
            return X

        n_pop, n_var = X.shape
        for i in range(n_pop):
            if np.random.rand() <= self.prob:
                for j in range(n_var):
                    if np.random.rand() < self.prob_var:
                        old = int(X[i, j])
                        # choose a new phase different from old (try limited times)
                        new = np.random.randint(0, self.n_phases)
                        tries = 0
                        while new == old and tries < 5:
                            new = np.random.randint(0, self.n_phases)
                            tries += 1
                        X[i, j] = new

        return X

class GeneticAlgorithm(Optimizer):
    """MCRpy optimizer adapter wrapping pymoo-based GA runs.

    This class expects DMCR to call `set_call_loss(call_loss)` after creation, where
    `call_loss` is a callable that accepts a `Microstructure` and returns a scalar loss.
    It then uses pymoo to search the space of integer phase assignments that minimize
    that loss using the existing sampling / mutation machinery in this module.
    """
    is_gradient_based = False
    is_vf_based = False
    is_sparse = False
    swaps_pixels = True

    def __init__(self, 
                 max_iter: int = 100, 
                 population_size: int = 50, 
                 conv_iter: int = 500,
                 callback: callable = None, 
                 loss: callable = None,
                 seed: int = None, 
                 n_phases: int = 2, # zero and one
                 tolerance: float = 0.00001,
                 use_multiphase: bool = False, 
                 mutation_rule: str = 'PM',
                 use_orientations: bool = False,
                 is_3D: bool = False,
                 **kwargs):
        if use_orientations:
            raise ValueError('This optimizer_type cannot solve for orientations.')
        self.max_iter = max_iter
        self.conv_iter = conv_iter #number of allowed iterations without improvement
        self.is_3D = is_3D
        self.population_size = population_size
        self.reconstruction_callback = callback
        self.seed = seed
        self.n_phases = int(n_phases)
        self.target_loss = tolerance
        self.use_multiphase = bool(use_multiphase)
        self.loss = loss
        self.tolerance = tolerance
        self.current_loss = np.inf
        self.loss_metadata = {}
        self.mutation_rule = mutation_rule

        # pymoo specific objects:
        self.problem = None
        self.sampling = None
        self.algorithm = None
        self.crossover = None
        self.mutation = None

    def set_up_pymoo(self):

        ms_shape = self.ms.spatial_shape
        n_elements = np.prod(ms_shape) #number of elements of each individual
        xl = np.zeros(n_elements) # lower bound for variables in function to be evaluated
        xu = np.full(n_elements, self.n_phases - 1) # upper bound for variables in function to be evaluated

        class OpimizationProblem(Problem):          
            def __init__(self,call_loss, use_multiphase):
                self.call_loss = call_loss
                self.use_multiphase = use_multiphase
                super().__init__(n_var=n_elements, # the number of variables for optimization problem is the size of the microstructure.
                                n_obj=1, # number of objective functions to be minimized 
                                n_constr=0, # number of constraints
                                type_var=int, # type of variables are ints
                                xl=xl,
                                xu=xu)

            def _evaluate(self, X, out, *args, **kwargs):
                fitness = []
                for individuum in X:
                    arr = individuum.astype(int).reshape(ms_shape)
                    try:
                        temp_ms = MutableMicrostructure(arr, use_multiphase=self.use_multiphase, trainable=False)
                        val = float(self.call_loss(temp_ms))
                    except Exception as e:
                        logging.debug(f'Error evaluating candidate: {e}')
                        val = np.inf
                    fitness.append(val)
                    out['F'] = np.array(fitness).reshape(-1,1)

        self.problem = OpimizationProblem(          
            call_loss=self.call_loss,
            use_multiphase=self.use_multiphase
            )

        self.sampling=DiverseRandomSampling()
        self.crossover = SBX(prob=0.9, eta=15,vtype=float, repair=RoundingRepair())
        self.mutation = PM(prob=0.5, eta=1,vtype=float, repair=RoundingRepair())

        self.algorithm = GA(
                pop_size=self.population_size,
                sampling=self.sampling,
                crossover=self.crossover,
                # Use PhaseBitflip to ensure mutated values remain in [0, n_phases-1]
                #mutation=PhaseBitflip(prob=0.5, prob_var=0.3, n_phases=n_phases), #seems to work well for very small examples
                mutation=self.mutation,
                eliminate_duplicates=True
                )
        
        # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
        self.algorithm.setup(self.problem, termination=NoTermination(), verbose=False)

    def optimize(self, ms: MutableMicrostructure, restart_from_niter: int = None):
        """Optimization loop."""

        self.n_iter = 0 if restart_from_niter is None else restart_from_niter
        self.iters_since_last_accept = 0
        self.ms = ms
        self.current_loss = self.call_loss(self.ms)

        self.set_up_pymoo() #setting up the genetic algorithm using pymoo library

        while self.n_iter < self.max_iter:
            if self.iters_since_last_accept >= self.conv_iter:
                logging.info('converged - no change since {self.iters_since_last_accept} iterations')
                break
            if self.current_loss <= self.tolerance:
                logging.info('reached tolerance')
                break
            self.step()
        else:
            logging.info('reached number of iterations')
        return self.n_iter

    def step(self):

        self.algorithm.next() # evaluate the next generation
        result = self.algorithm.result() # getting the results for the evaluated generation
        new_loss = float(result.F)
        X = result.X.copy()
        #current_best_ms = X[best_idx].copy()
        current_best_ms_arr = X.reshape(self.ms.spatial_shape)
        
            
        #new_loss = self.call_loss(self.ms)
        loss_amelioration = self.current_loss - new_loss
        if loss_amelioration > 0 :
            self.iters_since_last_accept = 0
            self.current_loss = new_loss
            self.ms = MutableMicrostructure(current_best_ms_arr, use_multiphase=self.use_multiphase, trainable=False)
        else:
            self.iters_since_last_accept += 1
        self.n_iter += 1
        self.reconstruction_callback(self.n_iter, self.current_loss, self.ms)
        return

    def _mutate(self):
        pass

def register() -> None:
    from mcrpy.src import optimizer_factory
    optimizer_factory.register("GeneticAlgorithm", GeneticAlgorithm)


