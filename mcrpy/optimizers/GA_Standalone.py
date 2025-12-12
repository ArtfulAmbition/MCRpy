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

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.sampling import Sampling
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
import os

# Configure TensorFlow/C++ logging and oneDNN before any TensorFlow import.
# - TF_CPP_MIN_LOG_LEVEL: 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
#   Setting to '3' hides INFO and WARNING, keeping only ERROR messages.
# - TF_ENABLE_ONEDNN_OPTS=0 disables oneDNN custom-op informational messages.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# Also set Python-side logger to ERROR for tensorflow logger (if TF is imported later)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import tortuosity descriptor from MCRpy (this will import TensorFlow internally)
from mcrpy.descriptors.Tortuosity import Tortuosity


class DiverseRandomSampling(Sampling):
    """Custom sampling to ensure phase diversity in initial population"""
    
    def _do(self, problem, n_samples, **kwargs):
        """Generate diverse initial population with mixed phases.
        Arg problem is a pymoo problem."""
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        
        n_phases = problem.xu[0] + 1  # Assuming all variables have same bounds. 
        # xu is the upper bound of allowed argument for the function to be optimized, 
        # that is 0 for one phase, 1 for two phases etc. Therefore the plus 1 is required.
        
        for i in range(n_samples):
            # Generate random microstructure with balanced phase distribution
            phases = np.random.randint(0, n_phases, problem.n_var)
            X[i] = phases
        
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
        """Evaluate fitness for population x. 
        The population x consists of individuals (= microstructures)"""
        fitness = np.zeros(len(x)) # initialize fitness
        
        for i, individual in enumerate(x):
            # Reshape to microstructure
            ms = individual.reshape(self.ms_shape).astype(int)
            
            # Compute tortuosity using MCRpy descriptor
            try:
                current_tort = float(self.descriptor(ms))
            except Exception as e:
                raise Exception('error in evaluation of tortuosity')
            
            # Loss: absolute difference from target
            loss = np.abs(current_tort - self.target_tortuosity)
            fitness[i] = loss
            self.eval_count += 1
        
        out["F"] = fitness


def run_ga_optimization(ms_shape, n_phases, target_tortuosity,
                       max_generations=1000, pop_size=150,
                       phase_of_interest=0, connectivity='sides',
                       method='DSPSM', direction=0,
                       voxel_dimension=(1, 1, 1),
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
        print("\n" + "="*70)
        print("MICROSTRUCTURE TORTUOSITY OPTIMIZATION (MCRpy)")
        print("="*70)
        print(f"Shape: {ms_shape}")
        print(f"Number of phases: {n_phases}")
        print(f"Target tortuosity: {target_tortuosity:.6f}")
        print(f"Phase of interest: {phase_of_interest}")
        print(f"Connectivity: {connectivity}")
        print(f"Method: {method}")
        print(f"Direction: {direction}")
        print(f"Population size: {pop_size}")
        print(f"Max generations: {max_generations}")
        print("="*70 + "\n")
    
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
    
    # Define algorithm
    algorithm = GA(
        pop_size=pop_size,
        sampling=DiverseRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # Callback for progress tracking
    best_loss_so_far = [float('inf')]
    
    def callback(algo):
        current_best = algo.pop.get("F").min()
        if current_best < best_loss_so_far[0]:
            best_loss_so_far[0] = current_best
            if verbose:
                print(f"Gen {algo.n_gen}: Loss improved to {current_best:.6f}")
    
    # Run optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', max_generations),
        seed=seed,
        verbose=False,
        callback=callback
    )
    
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
    
    if verbose:
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)
        print(f"Final loss: {best_loss:.6f}")
        print(f"Target tortuosity: {target_tortuosity:.6f}")
        print(f"Optimized tortuosity: {final_tort:.6f}")
        print(f"Tortuosity error: {np.abs(final_tort - target_tortuosity):.6f}")
        print(f"Generations: {res.algorithm.n_gen}")
        print(f"Total evaluations: {problem.eval_count}")
        print("="*70)
        print("\nOptimized microstructure:")
        print(optimized_ms)
        print()
    
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
    
    # # Example 1: Simple 2D microstructure - find phase 1 with target tortuosity
    # print("\n" + "#"*70)
    # print("# EXAMPLE 1: 2D (7x7), 2 phases, optimize PHASE 1, target tort=1.2")
    # print("#"*70)
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

    # Example 2: Simple 2D microstructure - find phase 2 with target tortuosity
    print("\n" + "#"*70)
    print("# EXAMPLE 1: 2D (7x7), 3 phases, optimize PHASE 2, target tort=1.2")
    print("#"*70)
    result_2d = run_ga_optimization(
        ms_shape=(3, 3),
        n_phases=2,
        target_tortuosity=5,
        max_generations=1000,
        pop_size=150,
        phase_of_interest=2, 
        connectivity='sides',
        method='DSPSM',
        direction=0,
        seed=42,
        verbose=True
    )

    # # Example 3: Simple 3D microstructure - find phase 2 with target tortuosity
    # print("\n" + "#"*70)
    # print("# EXAMPLE 1: 2D (7x7x7), 3 phases, optimize PHASE 2, target tort=1.2")
    # print("#"*70)
    # result_2d = run_ga_optimization(
    #     ms_shape=(7, 7, 7),
    #     n_phases=3,
    #     target_tortuosity=1.2,
    #     max_generations=150,
    #     pop_size=60,
    #     phase_of_interest=2, 
    #     connectivity='sides',
    #     method='DSPSM',
    #     direction=0,
    #     seed=42,
    #     verbose=True
    # )
