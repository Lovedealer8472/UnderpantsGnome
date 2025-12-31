"""
Multi-Objective Optimization - NSGA-II for parameter search.

Optimizes multiple objectives simultaneously:
1. ROI per day
2. Sharpe ratio
3. Win rate
4. Profit factor

Uses genetic algorithm to find Pareto-optimal configurations.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import copy


@dataclass
class Individual:
    """Represents a configuration (individual in genetic algorithm)."""
    genes: Dict[str, any]  # Parameter values
    fitness: Dict[str, float] = None  # Objective values
    rank: int = 0  # Pareto rank
    crowding_distance: float = 0.0  # Crowding distance
    
    def __repr__(self):
        return f"Individual(rank={self.rank}, fitness={self.fitness})"


class NSGAII:
    """Non-dominated Sorting Genetic Algorithm II for multi-objective optimization."""
    
    def __init__(
        self,
        search_space: Dict[str, List],
        objectives: List[str],
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        tournament_size: int = 3
    ):
        """
        Initialize NSGA-II optimizer.
        
        Args:
            search_space: Dictionary of parameter_name -> list of possible values
            objectives: List of objective names to maximize
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament for selection
        """
        self.search_space = search_space
        self.objectives = objectives
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
        self.population = []
        self.pareto_front_history = []
        
    def initialize_population(self) -> List[Individual]:
        """Create initial random population."""
        population = []
        
        for _ in range(self.population_size):
            genes = {}
            for param, values in self.search_space.items():
                genes[param] = random.choice(values)
            population.append(Individual(genes=genes))
        
        return population
    
    def evaluate_population(
        self,
        population: List[Individual],
        evaluation_func: Callable[[Dict], Dict[str, float]]
    ):
        """Evaluate fitness for all individuals."""
        for i, individual in enumerate(population):
            if individual.fitness is None:
                # Evaluate this configuration
                fitness = evaluation_func(individual.genes)
                individual.fitness = fitness
                
                if (i + 1) % 10 == 0:
                    print(f"  Evaluated {i+1}/{len(population)} individuals...")
    
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Fast non-dominated sorting algorithm.
        
        Returns:
            List of fronts (each front is a list of individuals)
        """
        # For each individual, calculate domination
        domination_count = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]
        
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i == j:
                    continue
                
                if self._dominates(p, q):
                    dominated_solutions[i].append(j)
                elif self._dominates(q, p):
                    domination_count[i] += 1
        
        # Create fronts
        fronts = [[]]
        for i, count in enumerate(domination_count):
            if count == 0:
                population[i].rank = 0
                fronts[0].append(population[i])
        
        # Build subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i, p in enumerate(population):
                if p in fronts[current_front]:
                    for j in dominated_solutions[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            population[j].rank = current_front + 1
                            next_front.append(population[j])
            
            current_front += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts
    
    def _dominates(self, p: Individual, q: Individual) -> bool:
        """Check if individual p dominates individual q."""
        if p.fitness is None or q.fitness is None:
            return False
        
        # p dominates q if p is no worse in all objectives and better in at least one
        better_in_at_least_one = False
        
        for obj in self.objectives:
            p_val = p.fitness.get(obj, 0)
            q_val = q.fitness.get(obj, 0)
            
            if p_val < q_val:  # p is worse in this objective
                return False
            if p_val > q_val:  # p is better in this objective
                better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def calculate_crowding_distance(self, front: List[Individual]):
        """Calculate crowding distance for individuals in a front."""
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        # Initialize distances
        for ind in front:
            ind.crowding_distance = 0
        
        # For each objective
        for obj in self.objectives:
            # Sort by objective value
            front_sorted = sorted(front, key=lambda x: x.fitness.get(obj, 0))
            
            # Boundary points have infinite distance
            front_sorted[0].crowding_distance = float('inf')
            front_sorted[-1].crowding_distance = float('inf')
            
            # Calculate distance for intermediate points
            obj_min = front_sorted[0].fitness.get(obj, 0)
            obj_max = front_sorted[-1].fitness.get(obj, 0)
            obj_range = obj_max - obj_min
            
            if obj_range > 0:
                for i in range(1, len(front_sorted) - 1):
                    distance = (
                        front_sorted[i + 1].fitness.get(obj, 0) -
                        front_sorted[i - 1].fitness.get(obj, 0)
                    ) / obj_range
                    front_sorted[i].crowding_distance += distance
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(population, self.tournament_size)
        
        # Sort by rank (lower is better), then by crowding distance (higher is better)
        tournament_sorted = sorted(
            tournament,
            key=lambda x: (x.rank, -x.crowding_distance)
        )
        
        return tournament_sorted[0]
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1_genes = {}
        child2_genes = {}
        
        for param in self.search_space.keys():
            if random.random() < 0.5:
                child1_genes[param] = parent1.genes[param]
                child2_genes[param] = parent2.genes[param]
            else:
                child1_genes[param] = parent2.genes[param]
                child2_genes[param] = parent1.genes[param]
        
        return Individual(genes=child1_genes), Individual(genes=child2_genes)
    
    def mutate(self, individual: Individual) -> Individual:
        """Mutate an individual."""
        mutated_genes = copy.deepcopy(individual.genes)
        
        for param, values in self.search_space.items():
            if random.random() < self.mutation_rate:
                mutated_genes[param] = random.choice(values)
        
        return Individual(genes=mutated_genes)
    
    def optimize(
        self,
        evaluation_func: Callable[[Dict], Dict[str, float]],
        progress_callback: Callable[[int, int, List[Individual]], None] = None
    ) -> List[Individual]:
        """
        Run NSGA-II optimization.
        
        Args:
            evaluation_func: Function that takes genes dict and returns fitness dict
            progress_callback: Optional callback(generation, total, pareto_front)
            
        Returns:
            Final Pareto front (list of non-dominated individuals)
        """
        print(f"\nInitializing NSGA-II with {self.population_size} individuals...")
        self.population = self.initialize_population()
        
        print("Evaluating initial population...")
        self.evaluate_population(self.population, evaluation_func)
        
        # Main evolution loop
        for gen in range(self.generations):
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{self.generations}")
            print(f"{'='*60}")
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(self.population)
            
            # Calculate crowding distance for each front
            for front in fronts:
                self.calculate_crowding_distance(front)
            
            # Store Pareto front
            pareto_front = fronts[0] if fronts else []
            self.pareto_front_history.append(copy.deepcopy(pareto_front))
            
            print(f"Pareto front size: {len(pareto_front)}")
            if pareto_front:
                print("Best individuals:")
                for i, ind in enumerate(pareto_front[:3]):
                    print(f"  {i+1}. {ind.fitness}")
            
            if progress_callback:
                progress_callback(gen + 1, self.generations, pareto_front)
            
            # Create offspring
            offspring = []
            while len(offspring) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(self.population)
                parent2 = self.tournament_selection(self.population)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                offspring.extend([child1, child2])
            
            # Trim to population size
            offspring = offspring[:self.population_size]
            
            # Evaluate offspring
            print(f"Evaluating offspring...")
            self.evaluate_population(offspring, evaluation_func)
            
            # Combine parent and offspring populations
            combined = self.population + offspring
            
            # Select next generation
            fronts = self.fast_non_dominated_sort(combined)
            next_population = []
            
            for front in fronts:
                if len(next_population) + len(front) <= self.population_size:
                    next_population.extend(front)
                else:
                    # Calculate crowding distance and sort
                    self.calculate_crowding_distance(front)
                    front_sorted = sorted(front, key=lambda x: -x.crowding_distance)
                    remaining = self.population_size - len(next_population)
                    next_population.extend(front_sorted[:remaining])
                    break
            
            self.population = next_population
        
        # Final Pareto front
        fronts = self.fast_non_dominated_sort(self.population)
        final_pareto_front = fronts[0] if fronts else []
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(f"Final Pareto front size: {len(final_pareto_front)}")
        
        return final_pareto_front
    
    def get_best_individual(self, pareto_front: List[Individual], weights: Dict[str, float] = None) -> Individual:
        """
        Get best individual from Pareto front using weighted sum.
        
        Args:
            pareto_front: List of non-dominated individuals
            weights: Optional weights for objectives (default: equal weights)
            
        Returns:
            Best individual according to weighted sum
        """
        if not pareto_front:
            return None
        
        if weights is None:
            weights = {obj: 1.0 / len(self.objectives) for obj in self.objectives}
        
        # Normalize objectives
        obj_mins = {obj: min(ind.fitness.get(obj, 0) for ind in pareto_front) for obj in self.objectives}
        obj_maxs = {obj: max(ind.fitness.get(obj, 0) for ind in pareto_front) for obj in self.objectives}
        
        best_score = -float('inf')
        best_individual = None
        
        for ind in pareto_front:
            score = 0
            for obj in self.objectives:
                val = ind.fitness.get(obj, 0)
                obj_range = obj_maxs[obj] - obj_mins[obj]
                if obj_range > 0:
                    normalized = (val - obj_mins[obj]) / obj_range
                else:
                    normalized = 0.5
                score += weights[obj] * normalized
            
            if score > best_score:
                best_score = score
                best_individual = ind
        
        return best_individual

