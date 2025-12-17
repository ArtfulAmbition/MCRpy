"""
   Copyright 2025 TU Dresden (Martin Sobczyk as Scientific Employee)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

disable_tf_warnings = True
if disable_tf_warnings:
    # Configure TensorFlow/C++ logging and oneDNN before any TensorFlow import.
    # - TF_CPP_MIN_LOG_LEVEL: 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
    #   Setting to '3' hides INFO and WARNING, keeping only ERROR messages.
    # - TF_ENABLE_ONEDNN_OPTS=0 disables oneDNN custom-op informational messages.
    import os
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import logging
import numpy as np
import tensorflow as tf

from mcrpy.optimizers.Optimizer import Optimizer
from mcrpy.src import optimizer_factory
from mcrpy.src.MutableMicrostructure import MutableMicrostructure

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from mpi4py import MPI
from mcrpy.src.log import mpi_logging

class MicrostructureReconstructionProblem(Problem):
    """Pymoo Problem Definition for Microstructure Reconstruction."""
    
    def __init__(self, ms_shape, loss_function, is_3D=False, use_mpi=False, n_phases=2):
        """
        Initialize the problem.
        
        Args:
            ms_shape: Shape of the microstructure
            loss_function: Function that computes loss/fitness
            is_3D: Whether problem is 3D
            use_mpi: Whether to use MPI parallelization
        """
        self.ms_shape = ms_shape
        self.loss_function = loss_function
        self.is_3D = is_3D
        self.eval_count = 0
        self.n_phases = n_phases
        self.use_mpi = use_mpi
        
        if self.use_mpi:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        
        n_var = int(np.prod(ms_shape))
        
        # Problem definition: Binary variables [0, 1]
        super().__init__(
            n_var=n_var, #number of arguments
            n_obj=1, #number of functions to minimize
            n_constr=0, #number of constraints
            type_var=float
        )
        
        # Set bounds explicitly as arrays for pymoo
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)*self.n_phases
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate population, optionally parallelized with MPI."""
        if self.use_mpi:
            self._evaluate_mpi(x, out)
        else:
            self._evaluate_serial(x, out)
    
    def _evaluate_serial(self, x, out):
        """Serial evaluation (single process)."""
        f = []
        
        for individual in x:
            # Convert to binary (0/1) as float64 to match Microstructure representation
            binary_individual = np.round(individual).astype(np.float64)
            binary_individual_reshaped = binary_individual.reshape(self.ms_shape)
            
           
            # Calculate Loss
            try:
                loss = float(self.loss_function(binary_individual_reshaped))
            except Exception:
                loss = np.inf
            
            # Gesamtfitness = Loss
            fitness = loss
            f.append(fitness)
            
            self.eval_count += 1
        
        out["F"] = np.array(f).reshape(-1, 1)
    
    def _evaluate_mpi(self, x, out):
        """MPI-parallelized evaluation across ranks."""
        n_individuals = len(x)
        
        # Distribute individuals to ranks
        individuals_per_rank = n_individuals // self.size
        remainder = n_individuals % self.size
        
        # Calculate start and end indices for this rank
        if self.rank < remainder:
            start_idx = self.rank * (individuals_per_rank + 1)
            end_idx = start_idx + individuals_per_rank + 1
        else:
            start_idx = remainder * (individuals_per_rank + 1) + (self.rank - remainder) * individuals_per_rank
            end_idx = start_idx + individuals_per_rank
        
        # Evaluate local individuals
        local_fitnesses = []
        for i in range(start_idx, end_idx):
            individual = x[i]
            binary_individual = np.round(individual).astype(np.float64)
            binary_individual_reshaped = binary_individual.reshape(self.ms_shape)
                       
            try:
                loss = float(self.loss_function(binary_individual_reshaped))
            except Exception:
                loss = np.inf
            
            fitness = loss
            local_fitnesses.append(fitness)
        
        # Gather all fitnesses to rank 0
        all_fitnesses = self.comm.allgather(local_fitnesses)
        
        # Flatten and assign to output
        f = []
        for rank_fitnesses in all_fitnesses:
            f.extend(rank_fitnesses)
        
        self.eval_count += n_individuals
        out["F"] = np.array(f).reshape(-1, 1)


class GeneticAlgorithm(Optimizer):
    """
    Genetic Algorithm Optimizer for Microstructure Reconstruction using pymoo.
    
    Features:
    - Binary representation of microstructure
    - Volume fraction constraints
    - Adaptive mutation and crossover
    - Multi-population support
    - Early stopping
    """
    
    is_gradient_based = False
    is_vf_based = False
    is_sparse = False
    swaps_pixels = True
    
    def __init__(self,
            max_iter: int = 100,
            conv_iter: int = 500,
            callback: callable = None,
            population_size: int = 50,
            mutation_rate: float = 0.1,
            crossover_rate: float = 0.9,
            mutation_eta: float = 20.0,
            crossover_eta: float = 15.0,
            seed: int = None,
            loss: callable = None,
            use_multiphase: bool = False,
            use_orientations: bool = False,
            is_3D: bool = False,
            target_loss: float = 1e-5,
            use_mpi: bool = False,
            **kwargs):
        """
        Initialize Genetic Algorithm Optimizer.
        
        Args:
            max_iter: Maximum number of generations
            conv_iter: Convergence iterations (stopping criterion)
            callback: Callback function after each iteration
            population_size: Size of population (default: 50)
            mutation_rate: Mutation probability (valid between 0 and 1, default: 0.1)
            crossover_rate: Crossover probability (valid between 0 and 1, default: 0.9)
            mutation_eta: Distribution index for mutation (higher = more local, default: 20)
            crossover_eta: Distribution index for crossover (higher = more local, default: 15)
            seed: Random seed for reproducibility
            loss: Loss function to minimize
            use_multiphase: Whether to handle multiphase materials
            use_orientations: Whether to handle orientations (not supported)
            is_3D: Whether problem is 3D
            target_loss: Target loss to stop optimization (default: 1e-5, set to 0 to disable)
            use_mpi: Whether to use MPI parallelization for population evaluation
        """
               
        if use_orientations:
            raise ValueError('GeneticAlgorithm cannot solve for orientations.')
        
        self.max_iter = max_iter
        self.is_3D = is_3D
        self.conv_iter = conv_iter
        self.reconstruction_callback = callback
        self.target_loss = target_loss
        self.use_mpi = use_mpi
        
        if self.use_mpi:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.mpi_size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.mpi_size = 1
        
        # GA-specific parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_eta = mutation_eta
        self.crossover_eta = crossover_eta
        self.seed = seed
        
        self.current_loss = np.inf
        self.loss = loss
        self.use_multiphase = use_multiphase
        self.last_tortuosity = None  # Store last computed tortuosity for logging
        self.loss_metadata = {}  # Store metadata from loss function
        
        # Default callback if none provided
        if self.reconstruction_callback is None:
            self.reconstruction_callback = lambda gen: None
        
        assert self.loss is not None, "loss function required"
        
        # History for convergence tracking
        self.fitness_history = []
        self.no_improve_count = 0
        
        msg = f"GeneticAlgorithm initialized with population_size={population_size}, max_iter={max_iter}"
        if self.use_mpi:
            msg += f" (MPI enabled: {self.mpi_size} ranks)"
        mpi_logging(msg)
        
    def optimize(self, ms, restart_from_niter: int = None):
        """
        Run genetic algorithm optimization.
        
        Args:
            ms: MutableMicrostructure object
            restart_from_niter: Iteration to restart from (if applicable)
        """
        
        mpi_logging(f"Starting GA optimization for microstructure shape {ms.xx.shape}")
        
        # Get microstructure as binary array
        ms_array = ms.xx.numpy() if isinstance(ms.xx, tf.Tensor) else np.array(ms.xx)
        ms_shape = ms_array.shape
        
        # Convert to continuous [0, 1] representation for GA
        x0 = ms_array.astype(float).flatten()
        
        # Define problem
        problem = MicrostructureReconstructionProblem(
            ms_shape=ms_shape,
            loss_function=self._evaluate_with_logging,
            is_3D=self.is_3D,
            use_mpi=self.use_mpi
        )
        
        # Define algorithm with adaptive parameters
        algorithm = GA(
            pop_size=self.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=self.crossover_rate, eta=self.crossover_eta),
            mutation=PM(eta=self.mutation_eta),
            eliminate_duplicates=True,
            seed=self.seed
        )
        
        # Run optimization
        mpi_logging(f"Starting optimization: {self.population_size} individuals, max {self.max_iter} generations")
        
        res = minimize(
            problem,
            algorithm,
            ('n_gen', self.max_iter),
            seed=self.seed,
            verbose=False,
            callback=self._pymoo_callback
        )
        
        # Extract best solution
        best_individual = np.round(res.X).astype(bool).reshape(ms_shape)
        best_loss = float(res.F[0])
        
        # In MPI mode, synchronize best solution across all ranks
        if self.use_mpi:
            # Each rank sends its best loss to rank 0
            all_losses = self.comm.gather(best_loss, root=0)
            all_solutions = self.comm.gather(best_individual.flatten().astype(float), root=0)
            
            if self.rank == 0:
                # Find globally best loss and corresponding solution
                global_best_idx = np.argmin(all_losses)
                global_best_loss = all_losses[global_best_idx]
                global_best_solution = all_solutions[global_best_idx]
                mpi_logging(f"Global best loss across {self.mpi_size} ranks (from rank {global_best_idx}): {global_best_loss:.6f}")
            else:
                global_best_loss = None
                global_best_solution = None
            
            # Broadcast global best to all ranks
            global_best_loss = self.comm.bcast(global_best_loss, root=0)
            global_best_solution = self.comm.bcast(global_best_solution, root=0)
            
            # Update local best with global best
            best_loss = global_best_loss
            best_individual = global_best_solution.reshape(ms_shape).astype(bool)
        
        self.current_loss = best_loss
        
        # Convert back to TensorFlow variable format and update
        # Reshape to match the internal x_shape
        best_individual_reshaped = best_individual.astype(np.float64).reshape(ms.x.shape)
        ms.x.assign(tf.constant(best_individual_reshaped, dtype=tf.float64))
        
        mpi_logging(f"GA optimization completed after {problem.eval_count} evaluations")
        mpi_logging(f"Final loss: {best_loss:.6f}")
        
        return res.algorithm.n_gen
    
    def _evaluate_with_logging(self, ms_array):
        """Evaluate microstructure with logging."""
        try:
            # Ensure it's a TensorFlow tensor
            if not isinstance(ms_array, tf.Tensor):
                ms_array = tf.constant(ms_array, dtype=tf.float32)

            # Suppress verbose logging from descriptor/DSPSM during GA evaluations
            # (otherwise logging fills terminal with repeated entries for each candidate)
            old_level = logging.root.level
            logging.root.setLevel(logging.WARNING)
            
            try:
                # Prefer using the pre-built `call_loss` (from loss_computation.make_call_loss)
                # which expects a `Microstructure` object. If it's available, wrap the
                # candidate array into a `MutableMicrostructure` and call it.
                if hasattr(self, 'call_loss') and callable(getattr(self, 'call_loss')):
                    try:
                        # Convert tensor to numpy array
                        arr = ms_array.numpy().astype(np.float64)

                        # If array has extra singleton dimensions (e.g., (1,20,20,1,1)),
                        # squeeze them until we have 2D or 3D, which MutableMicrostructure expects.
                        squeezed = np.squeeze(arr)
                        if squeezed.ndim not in {2, 3}:
                            squeezed = squeezed.reshape(-1) if squeezed.size > 0 else squeezed
                        arr_use = squeezed

                        temp_ms = MutableMicrostructure(arr_use)
                        val = self.call_loss(temp_ms)
                        return float(val)
                    except Exception as e:
                        mpi_logging(f"[GA debug] call_loss wrapper failed: {e}")
                        pass

                loss = float(self.loss(ms_array))
                return loss
            finally:
                # Restore logging level
                logging.root.setLevel(old_level)
        except Exception as e:
            logging.warning(f"Loss evaluation failed: {e}")
            return np.inf
    
    def _pymoo_callback(self, algorithm):
        """Callback function called by pymoo after each generation."""
        
        # Get best fitness in current population
        best_fitness = float(algorithm.pop.get("F").min())
        self.fitness_history.append(best_fitness)
        
        # Update current loss
        if best_fitness < self.current_loss:
            self.current_loss = best_fitness
            self.no_improve_count = 0
            
            if self.rank == 0:
                output = f"Gen {algorithm.n_gen}: Loss improved to {best_fitness:.6f}"
                
                # Add tortuosity info if available
                if hasattr(self, 'last_tort_value') and self.last_tort_value[0] is not None:
                    tort_value = self.last_tort_value[0]
                    goal_tort = getattr(self, 'goal_tort_value', None)
                    if goal_tort is not None:
                        output += f" | Tortuosity: {tort_value:.6f} (target: {goal_tort:.6f})"
                
                mpi_logging(output)
        else:
            self.no_improve_count += 1
        
        # Call user callback
        if self.reconstruction_callback is not None:
            try:
                self.reconstruction_callback(algorithm.n_gen)
            except Exception as e:
                logging.warning(f"Callback failed: {e}")
        
        # Stop if target loss is achieved (only if target_loss > 0, else skip)
        if self.target_loss > 0 and best_fitness <= self.target_loss:
            mpi_logging(f"Target loss of {self.target_loss} achieved.")
            algorithm.termination.force_termination = True
            return
        
        # Early stopping if no improvement
        if self.no_improve_count >= self.conv_iter:
            mpi_logging(f"Early stopping: No improvement for {self.conv_iter} generations")
            algorithm.termination.force_termination = True
        
        # Log progress
        if algorithm.n_gen % max(1, self.max_iter // 10) == 0:
            mpi_logging(f"Generation {algorithm.n_gen}/{self.max_iter}: "
                         f"Best fitness = {best_fitness:.6f}")


def register() -> None:
    """Register GeneticAlgorithm in optimizer factory."""
    optimizer_factory.register("GeneticAlgorithm", GeneticAlgorithm)


# Example usage
if __name__ == "__main__":
    import numpy as np
    from mcrpy.src.MutableMicrostructure import MutableMicrostructure
    from mcrpy.descriptors.Tortuosity import Tortuosity

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Get MPI rank for conditional printing
    try:
        rank = MPI.COMM_WORLD.Get_rank()
    except:
        rank = 0
      
    # Define microstructure shape
    ms_shape = (7, 7)
    
    # Create descriptor for tortuosity
    singlephase_descriptor = Tortuosity.make_singlephase_descriptor()
    
    # Prescribe target tortuosity value directly
    goal_tort = 3.5
    
    if rank == 0:
        print('='*60)
        print(f'Microstructure shape: {ms_shape}')
        print(f'Target tortuosity value: {goal_tort:.6f}')
        print('='*60)

    # Track last computed tortuosity value
    last_tort = [None]  # Use list to allow modification in nested function
    
    def loss_function(ms_array):
        """
        Loss function: minimize difference between current and goal tortuosity.
        
        Args:
            ms_array: Microstructure array (flattened or shaped, can be float [0,1])
        
        Returns:
            float: Loss value (L2 norm of tortuosity difference)
        """
        try:
            # Ensure it's a binary/boolean microstructure
            ms_binary = np.round(ms_array).astype(bool) if ms_array.dtype != bool else ms_array
            
            # Ensure proper shape
            if ms_binary.ndim == 1:
                ms_binary = ms_binary.reshape(ms_shape)
            
            # Compute tortuosity
            current_tort = singlephase_descriptor(ms_binary)
            last_tort[0] = current_tort  # Store for logging
            
            # Loss is difference from goal
            loss = float(np.abs(current_tort - goal_tort))
            return loss
        except Exception as e:
            # Return large penalty if computation fails
            return 1e6
    
    # Create initial microstructure (random)
    start_ms = np.random.random(ms_shape)
    
    if rank == 0:
        print(f'Initial loss: {loss_function(start_ms):.6f}')
        print(f'Initial tortuosity: {last_tort[0]:.6f}')
        print()
    
    # Create MutableMicrostructure wrapper
    mm = MutableMicrostructure(start_ms)
    
    # Custom callback to print improvements with tortuosity
    def custom_callback(gen):
        """Custom callback to print loss and tortuosity when improving."""
        pass  # Improvements will be logged by GA callback
    
    # Create and run GA optimizer
    ga = GeneticAlgorithm(
        max_iter=300,
        population_size=150,
        loss=loss_function,
        is_3D=False,
        target_loss=0.05,  # Stop when loss < 0.05
        use_mpi=True,
        mutation_rate = 0.1,
        callback=custom_callback
    )
    
    # Store reference to last_tort in GA for callback access
    ga.last_tort_value = last_tort
    ga.goal_tort_value = goal_tort
    
    # Run optimization
    result = ga.optimize(mm)

    # Print results (rank 0 only)
    if rank == 0:
        # Get optimized microstructure
        optimized_ms = mm.xx
        
        # Ensure it's binary
        optimized_ms_binary = np.round(optimized_ms).astype(bool)
        
        print('='*60)
        print('Optimization Results:')
        print('='*60)
        print(f'Initial loss: {loss_function(start_ms):.6f}')
        print(f'Final loss: {ga.current_loss:.6f}')
        print(f'Generations run: {len(ga.fitness_history)}')
        
        # Compute optimized tortuosity
        optimized_tort = singlephase_descriptor(optimized_ms_binary)
        print()
        print(f'Target tortuosity: {goal_tort:.6f}')
        print(f'Optimized tortuosity: {optimized_tort:.6f}')
        print(f'Tortuosity error: {np.abs(optimized_tort - goal_tort):.6f}')
        print('='*60)
        print(optimized_ms_binary.reshape(ms_shape).astype(int))