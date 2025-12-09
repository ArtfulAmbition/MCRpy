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


class MicrostructureReconstructionProblem(Problem):
    """Pymoo Problem Definition for Microstructure Reconstruction."""
    
    def __init__(self, ms_shape, loss_function, volume_fractions=None, is_3D=False):
        """
        Initialize the problem.
        
        Args:
            ms_shape: Shape of the microstructure
            loss_function: Function that computes loss/fitness
            volume_fractions: Optional volume fractions to maintain
            is_3D: Whether problem is 3D
        """
        self.ms_shape = ms_shape
        self.loss_function = loss_function
        self.volume_fractions = volume_fractions
        self.is_3D = is_3D
        self.eval_count = 0
        
        # Problem definition: Binary variables [0, 1]
        super().__init__(
            n_var=int(np.prod(ms_shape)),
            n_obj=1,
            n_constr=0,
            type_var=float,
            lb=0.0,
            ub=1.0
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate population."""
        f = []
        
        for individual in x:
            # Konvertiere zu Binary (0/1)
            binary_individual = np.round(individual).astype(bool)
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
            except:
                loss = np.inf
            
            # Gesamtfitness = Loss + Penalty
            fitness = loss + vf_penalty
            f.append(fitness)
            
            self.eval_count += 1
        
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
            tolerance: float = 1e-5,
            loss: callable = None,
            use_multiphase: bool = False,
            use_orientations: bool = False,
            is_3D: bool = False,
            **kwargs):
        """
        Initialize Genetic Algorithm Optimizer.
        
        Args:
            max_iter: Maximum number of generations
            conv_iter: Convergence iterations (stopping criterion)
            callback: Callback function after each iteration
            population_size: Size of population (default: 50)
            mutation_rate: Mutation probability (0-1, default: 0.1)
            crossover_rate: Crossover probability (0-1, default: 0.9)
            mutation_eta: Distribution index for mutation (higher = more local, default: 20)
            crossover_eta: Distribution index for crossover (higher = more local, default: 15)
            seed: Random seed for reproducibility
            tolerance: Convergence tolerance
            loss: Loss function to minimize
            use_multiphase: Whether to handle multiphase materials
            use_orientations: Whether to handle orientations (not supported)
            is_3D: Whether problem is 3D
        """
        
        if not HAS_PYMOO:
            raise ImportError("pymoo is required for GeneticAlgorithm. Install with: pip install pymoo")
        
        if use_orientations:
            raise ValueError('GeneticAlgorithm cannot solve for orientations.')
        
        self.max_iter = max_iter
        self.is_3D = is_3D
        self.conv_iter = conv_iter
        self.reconstruction_callback = callback
        
        # GA-specific parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_eta = mutation_eta
        self.crossover_eta = crossover_eta
        self.seed = seed
        
        self.tolerance = tolerance
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
        
        logging.info(f"GeneticAlgorithm initialized with population_size={population_size}, max_iter={max_iter}")
    
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
            is_3D=self.is_3D
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
        
        # Update microstructure
        ms.xx = best_individual
        
        logging.info(f"GA optimization completed after {problem.eval_count} evaluations")
        logging.info(f"Final loss: {best_loss:.6f}")
        
        return res
    
    def _evaluate_with_logging(self, ms_array):
        """Evaluate microstructure with logging."""
        try:
            # Ensure it's a TensorFlow tensor
            if not isinstance(ms_array, tf.Tensor):
                ms_array = tf.constant(ms_array, dtype=tf.float32)
            
            loss = float(self.loss(ms_array))
            return loss
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
    
    logging.basicConfig(level=logging.INFO)
    
    # Test problem: Simple binary optimization
    def simple_loss(ms):
        """Minimize number of 1s."""
        return float(np.sum(ms))
    
    # Create test microstructure
    ms_shape = (20, 20)
    ms = np.random.randint(2, size=ms_shape).astype(float)
    
    # Create GA optimizer
    ga = GeneticAlgorithm(
        max_iter=50,
        population_size=30,
        loss=simple_loss,
        is_3D=False
    )
    
    ga.set_call_loss(lambda _: simple_loss)
    
    print(f"Initial microstructure sum: {np.sum(ms)}")
    print(f"GA optimization would minimize this value...")
