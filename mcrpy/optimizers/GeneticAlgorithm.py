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

import logging
import numpy as np
import tensorflow as tf

from mcrpy.optimizers.Optimizer import Optimizer
from mcrpy.src import optimizer_factory
from mcrpy.src.MutableMicrostructure import MutableMicrostructure

try:
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination.default import DefaultMultiObjectiveTermination
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False
    logging.warning("pymoo not installed. Install with: pip install pymoo")

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    logging.debug("mpi4py not installed. GA will run serially. Install with: pip install mpi4py")


class MicrostructureReconstructionProblem(Problem):
    """Pymoo Problem Definition for Microstructure Reconstruction."""
    
    def __init__(self, ms_shape, loss_function, volume_fractions=None, is_3D=False, use_mpi=False):
        """
        Initialize the problem.
        
        Args:
            ms_shape: Shape of the microstructure
            loss_function: Function that computes loss/fitness
            volume_fractions: Optional volume fractions to maintain
            is_3D: Whether problem is 3D
            use_mpi: Whether to use MPI parallelization
        """
        self.ms_shape = ms_shape
        self.loss_function = loss_function
        self.volume_fractions = volume_fractions
        self.is_3D = is_3D
        self.eval_count = 0
        self.use_mpi = use_mpi and HAS_MPI
        
        if self.use_mpi:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            if self.rank == 0:
                logging.info(f"MPI enabled: {self.size} ranks available for parallelization")
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        
        n_var = int(np.prod(ms_shape))
        
        # Problem definition: Binary variables [0, 1]
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=0,
            type_var=float
        )
        
        # Set bounds explicitly as arrays for pymoo
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
    
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
            
            # Prüfe Volume Fractions
            if self.volume_fractions is not None:
                current_vf = np.sum(binary_individual_reshaped) / binary_individual_reshaped.size
                # Penalisiere Abweichung von Volume Fractions
                vf_penalty = 10.0 * np.abs(current_vf - self.volume_fractions[1])
            else:
                vf_penalty = 0.0
            
            # Berechne Loss
            try:
                loss = float(self.loss_function(binary_individual_reshaped))
            except Exception:
                loss = np.inf
            
            # Gesamtfitness = Loss + Penalty
            fitness = loss + vf_penalty
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
            
            if self.volume_fractions is not None:
                current_vf = np.sum(binary_individual_reshaped) / binary_individual_reshaped.size
                vf_penalty = 10.0 * np.abs(current_vf - self.volume_fractions[1])
            else:
                vf_penalty = 0.0
            
            try:
                loss = float(self.loss_function(binary_individual_reshaped))
            except Exception:
                loss = np.inf
            
            fitness = loss + vf_penalty
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
    is_vf_based = True
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
        
        if not HAS_PYMOO:
            raise ImportError("pymoo is required for GeneticAlgorithm. Install with: pip install pymoo")
        
        if use_orientations:
            raise ValueError('GeneticAlgorithm cannot solve for orientations.')
        
        self.max_iter = max_iter
        self.is_3D = is_3D
        self.conv_iter = conv_iter
        self.reconstruction_callback = callback
        self.target_loss = target_loss
        self.use_mpi = use_mpi and HAS_MPI
        
        if self.use_mpi:
            self.mpi_rank = MPI.COMM_WORLD.Get_rank()
            self.mpi_size = MPI.COMM_WORLD.Get_size()
        else:
            self.mpi_rank = 0
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
        self.volume_fractions = None
        self.use_multiphase = use_multiphase
        
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
        logging.info(msg)
    
    def set_volume_fractions(self, volume_fractions: np.ndarray):
        """Set volume fractions to maintain."""
        self.volume_fractions = volume_fractions
        logging.info(f"Volume fractions set: {volume_fractions}")
    
    def optimize(self, ms, restart_from_niter: int = None):
        """
        Run genetic algorithm optimization.
        
        Args:
            ms: MutableMicrostructure object
            restart_from_niter: Iteration to restart from (if applicable)
        """
        
        logging.info(f"Starting GA optimization for microstructure shape {ms.xx.shape}")
        
        # Get microstructure as binary array
        ms_array = ms.xx.numpy() if isinstance(ms.xx, tf.Tensor) else np.array(ms.xx)
        ms_shape = ms_array.shape
        
        # Convert to continuous [0, 1] representation for GA
        x0 = ms_array.astype(float).flatten()
        
        # Define problem
        problem = MicrostructureReconstructionProblem(
            ms_shape=ms_shape,
            loss_function=self._evaluate_with_logging,
            volume_fractions=self.volume_fractions,
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
        logging.info(f"Starting optimization: {self.population_size} individuals, max {self.max_iter} generations")
        
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
        
        self.current_loss = best_loss
        
        # Convert back to TensorFlow variable format and update
        # Reshape to match the internal x_shape
        best_individual_reshaped = best_individual.astype(np.float64).reshape(ms.x.shape)
        ms.x.assign(tf.constant(best_individual_reshaped, dtype=tf.float64))
        
        logging.info(f"GA optimization completed after {problem.eval_count} evaluations")
        logging.info(f"Final loss: {best_loss:.6f}")
        
        return res
    
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
                        logging.debug(f"[GA debug] call_loss wrapper failed: {e}")
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
            logging.debug(f"Gen {algorithm.n_gen}: Best fitness improved to {best_fitness:.6f}")
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
            logging.info(f"Target loss of {self.target_loss} achieved.")
            algorithm.termination.force_termination = True
            return
        
        # Early stopping if no improvement
        if self.no_improve_count >= self.conv_iter:
            logging.info(f"Early stopping: No improvement for {self.conv_iter} generations")
            algorithm.termination.force_termination = True
        
        # Log progress
        if algorithm.n_gen % max(1, self.max_iter // 10) == 0:
            logging.info(f"Generation {algorithm.n_gen}/{self.max_iter}: "
                         f"Best fitness = {best_fitness:.6f}")


def register() -> None:
    """Register GeneticAlgorithm in optimizer factory."""
    optimizer_factory.register("GeneticAlgorithm", GeneticAlgorithm)


# Example usage
if __name__ == "__main__":
    import numpy as np
    from mcrpy.src.MutableMicrostructure import MutableMicrostructure
    
    logging.basicConfig(level=logging.INFO)
    
    # Test problem: Simple binary optimization
    def simple_loss(ms):
        """Minimize number of 1s."""
        return float(np.sum(ms))
    
    import os
    from mcrpy.descriptors.Tortuosity import Tortuosity

    folder = '/home/sobczyk/Dokumente/MCRpy/example_microstructures' 
    minimal_example_ms = os.path.join(folder,'Holzer2020_Fine_Zoom0.33_Size60.npy')

    # ms = np.load(minimal_example_ms)
    # ms = ms[:,:,-2:-1]
    # ms = np.zeros((5,5))
    # ms[1,2] = 1
    # ms[2,1] = 1
    # ms[2,2] = 1
    # ms[2,3] = 1
    # ms[3,2] = 1
    ms = np.random.randint(0, 2, size=(20,20,20))
    ms_shape = ms.shape

    singlephase_descriptor = Tortuosity.make_singlephase_descriptor()
    mean_tort = singlephase_descriptor(ms)
    #mean_tort = 20.0 #example
    print(f'goal tort: {mean_tort}')

    def simple_loss(current_ms):
        return np.linalg.norm(singlephase_descriptor(current_ms) - mean_tort)
    
    # Create test microstructure with matching volume fraction
    # (ensures non-zero tortuosity descriptor for random candidates)
    ms_shape = ms.shape
    initial_ms = np.random.random(ms_shape)
    
    # Create MutableMicrostructure wrapper
    mm = MutableMicrostructure(initial_ms)
    
    # Create and run GA optimizer (target_loss=0 means no early exit on target)
    ga = GeneticAlgorithm(
        max_iter=20,
        population_size=20,
        loss=simple_loss,
        is_3D=True,
        use_mpi=True
    )
    
    # Run optimization
    result = ga.optimize(mm)
    print(f'result: {result}')
    # Get optimized microstructure
    optimized_ms = mm.xx
    print(f"\nOptimization Results:")
    print(f"  Initial loss: {simple_loss(initial_ms)}")
    print(f"  Optimized loss: {simple_loss(optimized_ms)}")
    print(f"  Final loss: {ga.current_loss:.6f}")
    print(f"  Fitness history: {ga.fitness_history[:5]}... (last: {ga.fitness_history[-1]:.6f})")
    print(f'\noriginal ms:\n {ms}\n')
    # print(f'optimized ms:\n {optimized_ms.numpy().reshape(ms_shape)}')
    print(f'optimized ms:\n {optimized_ms.reshape(ms.shape)}')

    print(f'tort value of optimized structure: {singlephase_descriptor(optimized_ms)}')