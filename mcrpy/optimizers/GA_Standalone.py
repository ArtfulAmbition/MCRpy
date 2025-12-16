"""
Minimalistic Genetic Algorithm for Microstructure Tortuosity Optimization

This script finds microstructures with prescribed tortuosity using a genetic algorithm.
Uses the Tortuosity descriptor from MCRpy.

Supports:
  - 2D and 3D microstructures
  - Any number of phases (0 to n_phases)
  - Flexible shape prescription
  - Customizable connectivity and method (DSPSM/SSPSM)
"""


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
from pymoo.operators.mutation.bitflip import BitflipMutation 
from pymoo.core.mutation import Mutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.sampling import Sampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
import logging
import warnings
from mcrpy.descriptors.Tortuosity import Tortuosity

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
            ny, nx = ms.shape
            other_direction = other_directions[0]
            rand_int = np.random.randint(0, ms.shape[other_direction])
            if direction == 0:  # x-direction: set full row to connect left to right
                ms[rand_int, :] = phase
            elif direction == 1:  # y-direction: set full column to connect top to bottom
                ms[:, rand_int] = phase
        elif dim == 3:
            nz, ny, nx = ms.shape
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


class MicrostructureOptimizationProblem(Problem):
    """Genetic Algorithm Problem: Find microstructure with target tortuosity"""
    
    def __init__(self, ms_shape, n_phases, target_tortuosity, 
                 phase_of_interest=0, connectivity='sides', method='DSPSM',
                 direction=0, voxel_dimension=(1, 1, 1)):
        """
        Args:
            ms_shape: Shape of microstructure (tuple: (ny, nx) for 2D or (nz, ny, nx) for 3D)
            n_phases: Number of phases (integer phases 0 to n_phases-1)
            target_tortuosity: Target tortuosity value
            phase_of_interest: Phase to analyze for tortuosity (default 0)
            connectivity: Connectivity type ('sides', 'edges', 'corners')
            method: Tortuosity method ('DSPSM' or 'SSPSM')
            direction: Direction of tortuosity analysis (0=x, 1=y, 2=z)
            voxel_dimension: Voxel size tuple
        """
        self.ms_shape = ms_shape
        self.n_elements = np.prod(ms_shape)
        self.n_phases = n_phases
        self.target_tortuosity = target_tortuosity
        self.phase_of_interest = phase_of_interest
        
        # Create tortuosity descriptor
        self.descriptor = Tortuosity.make_singlephase_descriptor(
            connectivity=connectivity,
            method=method,
            direction=direction,
            phase_of_interest=phase_of_interest,
            voxel_dimension=voxel_dimension
        )
        
        self.eval_count = 0
        
        # Create bounds arrays for each variable
        xl = np.zeros(self.n_elements) # lower bound for variables in function to be evaluated
        xu = np.full(self.n_elements, n_phases - 1) # upper bound for variables in function to be evaluated
        
        super().__init__(
            n_var=self.n_elements, # the number of variables for optimization problem is the size of the microstructure.
            n_obj=1, # number of objective functions to be minimized (here: only one, optimize tortuosity)
            n_constr=0, # number of constraints
            type_var=int, # type of variables are ints (= phase numbers)
            xl=xl,
            xu=xu
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate fitness for population x using MPI parallelization."""
           
        # print(f'rank {rank}: x in evaluate:\n {x}')

        # Split the population across MPI ranks
        chunks = np.array_split(x, mpi_size)
        local_chunk = chunks[rank]
        # print(f'rank {rank}: local_chunk:\n {local_chunk}')
        
        local_fitness = np.zeros(len(local_chunk))
        for i, individual in enumerate(local_chunk):
            # Reshape to microstructure
            ms = individual.reshape(self.ms_shape).astype(int)
            
            # Compute tortuosity using MCRpy descriptor
            try:
                current_tort = float(self.descriptor(ms))
            except Exception as e:
                raise Exception('error in evaluation of tortuosity')
            
            # Loss: absolute difference from target
            loss = np.abs(current_tort - self.target_tortuosity)
            local_fitness[i] = loss
            self.eval_count += 1
        
        # Gather fitness from all ranks
        #print(f'process rank {rank} evalutated local fitness {local_fitness}.')
        all_fitness = comm.allgather(local_fitness)
        fitness = np.concatenate(all_fitness)
        #mpi_logging(f'process rank {rank} evalutated total fitness {fitness}.')
        
        out["F"] = fitness


def run_ga_optimization(ms_shape, n_phases, target_tortuosity,
                       max_generations=1000, pop_size=150,
                       phase_of_interest=0, connectivity='sides',
                       method='DSPSM', direction=0,
                       voxel_dimension=(1, 1, 1),
                       stop_loss_tol: float = None,
                       seed=None, verbose=True):
    """
    Run genetic algorithm to optimize microstructure tortuosity
    
    Args:
        ms_shape: Shape of microstructure ((ny, nx) for 2D or (nz, ny, nx) for 3D)
        n_phases: Number of phases (integer phases 0 to n_phases-1)
        target_tortuosity: Target tortuosity value
        max_generations: Maximum GA generations
        pop_size: Population size
        phase_of_interest: Phase ID to optimize for (default 0)
        connectivity: Connectivity type ('sides', 'edges', 'corners')
        method: Tortuosity method ('DSPSM' or 'SSPSM')
        direction: Direction of analysis (0=x, 1=y, 2=z)
        voxel_dimension: Voxel size tuple
        seed: Random seed for reproducibility
        verbose: Print progress information
        
    Returns:
        Dictionary with:
        - 'optimized_ms': Best microstructure found (numpy array)
        - 'final_loss': Best loss achieved
        - 'final_tort': Actual tortuosity of optimized microstructure
        - 'generations': Number of generations run
        - 'evaluations': Total function evaluations
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    if verbose:
        mpi_logging("\n" + "="*70)
        mpi_logging("MICROSTRUCTURE TORTUOSITY OPTIMIZATION (MCRpy)")
        mpi_logging("="*70)
        mpi_logging(f"Shape: {ms_shape}")
        mpi_logging(f"Number of phases: {n_phases}")
        mpi_logging(f"Target tortuosity: {target_tortuosity:.6f}")
        mpi_logging(f"Phase of interest: {phase_of_interest}")
        mpi_logging(f"Connectivity: {connectivity}")
        mpi_logging(f"Method: {method}")
        mpi_logging(f"Direction: {direction}")
        mpi_logging(f"Population size: {pop_size}")
        mpi_logging(f"Max generations: {max_generations}")
        mpi_logging("="*70 + "\n")
    
    # Configure logging to suppress intermediate messages
    logging.basicConfig(level=logging.CRITICAL, force=True)
    
    # Define problem
    problem = MicrostructureOptimizationProblem(
        ms_shape=ms_shape,
        n_phases=n_phases,
        target_tortuosity=target_tortuosity,
        phase_of_interest=phase_of_interest,
        connectivity=connectivity,
        method=method,
        direction=direction,
        voxel_dimension=voxel_dimension
    )
    
    # Define algorithm with tuned parameters for discrete optimization
    # For discrete (integer) problems, we need higher mutation rates
    algorithm = GA(
        pop_size=pop_size,
        sampling=DiverseRandomSampling(),
        crossover=SBX(prob=0.9, eta=15,vtype=float, repair=RoundingRepair()),
        # Use PhaseBitflip to ensure mutated values remain in [0, n_phases-1]
        #mutation=PhaseBitflip(prob=0.5, prob_var=0.3, n_phases=n_phases), #seems to work well for very small examples
        mutation=PM(prob=0.5, eta=1,vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True
    )
    
    # Callback for progress tracking
    best_loss_so_far = [float('inf')]
    
    stop_data = {}

    def callback(algorithm):
        current_pop = algorithm.pop
        F = current_pop.get("F")
        current_best = float(np.min(F))
        if current_best < best_loss_so_far[0]:
            best_loss_so_far[0] = current_best
            if verbose:
                mpi_logging(f"Gen {algorithm.n_gen}: Loss improved to {current_best:.6f}")
        else:
            if verbose:
                mpi_logging(f"Gen {algorithm.n_gen}: no improvement.",end='\r')


        # Early stop if a loss threshold is provided and reached
        if stop_loss_tol is not None and current_best <= stop_loss_tol:
            best_idx = int(np.argmin(F))
            Xpop = current_pop.get("X")
            stop_data['X'] = Xpop[best_idx].copy()
            stop_data['F'] = float(F[best_idx])
            stop_data['n_gen'] = int(algorithm.n_gen)
            raise StopIteration("early stop: loss below threshold")
        
    # Run optimization (catch StopIteration from early-stop callback)
    try:
        res = minimize(
            problem,
            algorithm,
            ('n_gen', max_generations),
            seed=seed,
            verbose=False,
            callback=callback
        )
    except StopIteration:
        # Build a minimal result-like object from stop_data
        class SimpleRes:
            pass
        res = SimpleRes()
        res.X = stop_data['X']
        res.F = np.array([stop_data['F']])
        res.algorithm = type('A', (), {'n_gen': stop_data.get('n_gen', 0)})()
    
    # Extract best solution
    best_individual = res.X
    best_loss = res.F[0]
    
    # Reshape and compute final tortuosity
    optimized_ms = best_individual.reshape(ms_shape).astype(int)
    
    # Re-compute final tortuosity using descriptor
    descriptor = Tortuosity.make_singlephase_descriptor(
        connectivity=connectivity,
        method=method,
        direction=direction,
        phase_of_interest=phase_of_interest,
        voxel_dimension=voxel_dimension
    )
    final_tort = float(descriptor(optimized_ms))

    total_eval_count = comm.reduce(problem.eval_count, op=MPI.SUM, root=0)
    
    if verbose:
        mpi_logging("\n" + "="*70)
        mpi_logging("OPTIMIZATION RESULTS")
        mpi_logging("="*70)
        mpi_logging(f"Final loss: {best_loss:.6f}")
        mpi_logging(f"Target tortuosity: {target_tortuosity:.6f}")
        mpi_logging(f"Optimized tortuosity: {final_tort:.6f}")
        mpi_logging(f'direction {direction}, phase {phase_of_interest}')
        mpi_logging(f"Tortuosity error: {np.abs(final_tort - target_tortuosity):.6f}")
        mpi_logging(f"Generations: {res.algorithm.n_gen}")
        mpi_logging(f"Total evaluations: {total_eval_count}")
        mpi_logging("="*70)
        mpi_logging("\nOptimized microstructure:")
        mpi_logging(optimized_ms)
        mpi_logging()
    
    return {
        'optimized_ms': optimized_ms,
        'final_loss': best_loss,
        'final_tort': final_tort,
        'generations': res.algorithm.n_gen,
        'evaluations': problem.eval_count
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
# Configure logging for standalone execution
    import os
    # pid = os.getpid()
    # print("pid: ",pid,flush=True)
    # if comm.rank == 0:
    #     input("Enter")
    # comm.Barrier()
    #exit()
    log_dir = './GA_Standalone_logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'GA_Standalone.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Override any previous basicConfig
    )
    logging.info("="*60)
    mpi_logging(f'TORTUOSITY DESCRIPTOR - STANDALONE EXECUTION')
    logging.info("="*60)
    # Suppress Matplotlib logging
    #logging.getLogger('matplotlib').setLevel(logging.WARNING)


    # # Example 1: Simple 2D microstructure - find phase 1 with target tortuosity
    # mpi_logging("\n" + "#"*70)
    # mpi_logging("# EXAMPLE 1: 2D (7x7), 2 phases, optimize PHASE 1, target tort=1.2")
    # mpi_logging("#"*70)
    # result_2d = run_ga_optimization(
    #     ms_shape=(7, 7),
    #     n_phases=2,
    #     target_tortuosity=1.2,
    #     max_generations=150,
    #     pop_size=60,
    #     phase_of_interest=1,  # Optimize phase 1, not phase 0
    #     connectivity='sides',
    #     method='DSPSM',
    #     direction=0,
    #     seed=42,
    #     verbose=True
    # )

    # Example: 2D microstructure with target tortuosity 2.5
    # Achievable pattern: alternating phase 0 stripes (as shown in your 4x4 example)
    mpi_logging("#"*70)
    mpi_logging("# EXAMPLE: 2D (7x7), 2 phases, optimize PHASE 0, target tort=2.5")
    mpi_logging("# (Based on proven achievable pattern from 4x4)")
    mpi_logging("#"*70)
    result_2d = run_ga_optimization(
        ms_shape=(70, 70, 70),
        n_phases=2,
        target_tortuosity=5,
        max_generations=5,
        pop_size=20,
        phase_of_interest=0, 
        connectivity='corners',
        method='SSPSM',
        direction=0,
        stop_loss_tol = 1e-2,
        seed=np.random.randint(1, 1000),
        verbose=True
    )

    # # Example 3: Simple 3D microstructure - find phase 2 with target tortuosity
    # mpi_logging("\n" + "#"*70)
    # mpi_logging("# EXAMPLE 1: 2D (7x7x7), 3 phases, optimize PHASE 2, target tort=1.2")
    # mpi_logging("#"*70)
    # result_2d = run_ga_optimization(
    #     ms_shape=(7, 7, 7),
    #     n_phases=3,
    #     target_tortuosity=10,
    #     max_generations=1000,
    #     pop_size=150,
    #     phase_of_interest=1, 
    #     connectivity='sides',
    #     method='DSPSM',
    #     direction=0,
    #     stop_loss_tol = 1e-4,
    #     #seed=42,
    #     verbose=True
    # )
