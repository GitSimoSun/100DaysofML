import numpy as np
import matplotlib.pyplot as plt

class Selection:
    """Handles different selection methods in genetic algorithms."""

    def __init__(self, method: str, y: np.ndarray, params: dict):
        self.method = method
        self.y = y
        self.k = params.get("tournament_k", 5)

    def select(self):
        if self.method == "truncation":
            return self.truncation_selection()
        elif self.method == "tournament":
            return self.tournament_selection()
        elif self.method == "roulette-wheel":
            return self.roulette_wheel_selection()
        else:
            raise ValueError("Invalid selection method")

    def truncation_selection(self):
        sorted_indices = np.argsort(self.y)
        selected_indices = np.random.choice(sorted_indices[:self.k], size=(len(self.y), 2))
        return selected_indices

    def tournament_selection(self):
        def get_parent():
            indices = np.random.choice(len(self.y), self.k, replace=False)
            return indices[np.argmin(self.y[indices])]
        return [[get_parent(), get_parent()] for _ in range(len(self.y))]

    def roulette_wheel_selection(self):
        normalized_fitness = self.y / self.y.sum()
        cumulative_probs = np.cumsum(normalized_fitness)

        def get_parent():
            r = np.random.rand()
            return np.searchsorted(cumulative_probs, r)

        return [[get_parent(), get_parent()] for _ in range(len(self.y))]


class Crossover:
    """Handles different crossover methods for generating new offspring."""

    def __init__(self, method: str, a: np.ndarray, b: np.ndarray, params: dict):
        self.method = method
        self.a = a
        self.b = b
        self.alpha = params.get("alpha", 0.5)

    def crossover(self):
        if self.method == "single-point":
            return self.single_point_crossover()
        elif self.method == "two-points":
            return self.two_points_crossover()
        elif self.method == "uniform":
            return self.uniform_crossover()
        elif self.method == "interpolation":
            return self.interpolation_crossover(self.alpha)
        else:
            raise ValueError("Invalid crossover method")

    def single_point_crossover(self):
        i = np.random.randint(1, len(self.a))
        return np.concatenate((self.a[:i], self.b[i:]))

    def two_points_crossover(self):
        i, j = sorted(np.random.choice(len(self.a), 2, replace=False))
        return np.concatenate((self.a[:i], self.b[i:j], self.a[j:]))

    def uniform_crossover(self):
        return np.where(np.random.rand(len(self.a)) < 0.5, self.a, self.b)

    def interpolation_crossover(self, alpha):
        return (1 - alpha) * self.a + alpha * self.b


class Mutation:
    """Handles different mutation methods for modifying offspring."""

    def __init__(self, method: str, child: np.ndarray, params: dict):
        self.method = method
        self.child = child
        self.alpha = params.get("alpha", 0.1)
        self.sigma = params.get("sigma", 0.1)

    def mutate(self):
        if self.method == "bitwise":
            return self.bitwise_mutation(self.alpha)
        elif self.method == "gaussian":
            return self.gaussian_mutation(self.sigma)
        else:
            raise ValueError("Invalid mutation method")

    def bitwise_mutation(self, alpha):
        return np.array([(1 - gene) if np.random.rand() < alpha else gene for gene in self.child])

    def gaussian_mutation(self, sigma):
        return self.child + np.random.randn(len(self.child)) * sigma


def rosenbrock(individual, a=1, b=100):
    x, y = individual
    return (a - x)**2 + b * (y - x**2)**2


def genetic_algorithm(f, population, selection_method, crossover_method, mutation_method, params, k_max=20, visualize=False):
    """Runs a genetic algorithm on a population with specified fitness function and methods.
    
    Args:
        f (function): Fitness function.
        population (np.ndarray): Initial population.
        selection_method (str): Selection method name.
        crossover_method (str): Crossover method name.
        mutation_method (str): Mutation method name.
        params (dict): Additional parameters.
        k_max (int): Number of generations.
        visualize (bool): Whether to visualize the population evolution.
        
    Returns:
        np.ndarray: Best individual found.
    """
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 6))
        X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
        Z = f([X, Y])
        ax.contour(X, Y, Z, levels=50, cmap="viridis")
        ax.set_title("Population Evolution Across Generations")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        scatter_plot = None

    for k in range(k_max):
        fitness_values = np.array([f(ind) for ind in population])
        selection = Selection(selection_method, fitness_values, params)
        parent_pairs = selection.select()

        children = []
        for indices in parent_pairs:
            parent1, parent2 = population[indices[0]], population[indices[1]]
            crossover = Crossover(crossover_method, parent1, parent2, params)
            child = crossover.crossover()
            mutation = Mutation(mutation_method, child, params)
            child = mutation.mutate()
            children.append(child)

        population = np.array(children)

        if visualize:
            if scatter_plot:
                scatter_plot.remove()
            scatter_plot = ax.scatter(population[:, 0], population[:, 1], color='red', s=10)
            plt.pause(0.3)

    if visualize:
        plt.show()

    best_individual = population[np.argmin([f(ind) for ind in population])]
    return best_individual


# Set genetic algorithm parameters
population_size = 30
num_generations = 100
selection_method = "tournament"
crossover_method = "interpolation"
mutation_method = "gaussian"
params = {
    "tournament_k": 5,
    "alpha": 0.5,
    "sigma": 0.05
}

# Initialize population
population = np.random.uniform(-5, 5, (population_size, 2))

# Run the genetic algorithm
best_solution = genetic_algorithm(
    rosenbrock,
    population,
    selection_method,
    crossover_method,
    mutation_method,
    params,
    k_max=num_generations,
    visualize=True  # Set to False if you don't want to visualize
)

print("Best solution found:", best_solution)
print("Function value at best solution:", rosenbrock(best_solution))
