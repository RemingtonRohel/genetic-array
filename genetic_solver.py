import copy
import random
import numpy as np
from scipy.constants import speed_of_light as c
import matplotlib.pyplot as plt


class GeneticSolver:
    """Implements a genetic algorithm for optimizing the phase of each element in an antenna
    array to synthesize a beam pattern specified by a peak sidelobe level and passband ripple.

    This is an attempt at recreating the method of Boeringer et al. (2005)
    https://ieeexplore-ieee-org.cyber.usask.ca/document/1377611
    """
    def __init__(self, num_antennas, antenna_spacing, freq, pb_ripple, sb_gain, passband, transition_width, num_points,
                 pop_size):
        self._population = pop_size
        self._num_antennas = num_antennas
        # Symmetric, and fix the end antennas to 0 phase
        self._num_weights = int(np.ceil((num_antennas - 2) / 2))
        self._antenna_spacing = antenna_spacing
        self._freq = freq

        pb_lower_bound = -pb_ripple
        sb_upper_bound = sb_gain

        self._sine_space = (2 * np.arange(num_points) / (num_points - 1)) - 1

        self._antenna_positions = antenna_spacing * (np.arange(self._num_antennas) - (self._num_antennas - 1) / 2)
        self._steering_matrix = np.exp(2j * np.pi * freq / c *
                                       np.einsum('i,j->ji', self._antenna_positions, self._sine_space))

        angles = np.arcsin(self._sine_space) * 180 / np.pi
        upper_bounds = np.ones(angles.size) * sb_upper_bound
        upper_bounds[np.argwhere(angles > passband[0] - transition_width)] = np.inf
        # upper_bounds[np.argwhere(angles > passband[0])] = pb_upper_bound
        # upper_bounds[np.argwhere(angles > passband[1])] = np.inf
        upper_bounds[np.argwhere(angles > passband[1] + transition_width)] = sb_upper_bound
        self._upper_bound = upper_bounds

        lower_bounds = np.ones(angles.size) * -1 * np.inf
        lower_bounds[np.argwhere(angles > passband[0])] = pb_lower_bound
        lower_bounds[np.argwhere(angles > passband[1])] = -1 * np.inf
        self._lower_bound = lower_bounds

        # These parameters are for diversifying which parameters get alternated for each iteration
        self._parameter_orders = np.array([[0, 0, 0],
                                           [1, 0, 1],
                                           [0, 1, 0],
                                           [1, 1, 1],
                                           [1, 0, 0],
                                           [0, 0, 1],
                                           [1, 1, 0],
                                           [0, 1, 1]])
        self._param_cycle = [[0, 1, 2],
                             [1, 2, 0],
                             [2, 0, 1],
                             [2, 1, 0],
                             [1, 0, 2],
                             [0, 2, 1]]
        self._cycle_idx = 0

        # These parameters define the adaptiveness of the algorithm
        self._mutation_ranges = np.square(np.ones(10) * 1.5) / (1.5 ** 15)
        self._mutation_rates = np.linspace(0.02, 0.2, 10)

        self._crossovers = np.linspace(0, min(4, self._num_weights), 17)

        # Initial parameters, will get changed as the algorithm progresses
        self._mut_rate_idx = self._mutation_rates.size - 1
        self._mut_range_idx = self._mutation_ranges.size - 1
        self._cross_idx = 0

        # 1 for increasing, -1 for decreasing
        self._mut_rate_dir = -1
        self._mut_range_dir = -1
        self._cross_dir = 1

        self._scores = []
        self._weights = self.find_weights()

    def find_weights(self):
        """The top-level engine for the algorithm. All iteration is done in this function,
        which returns the final set of weights."""
        members = self.initialize_population()
        member_ff = self.calculate_far_field(members)
        member_scores = self.cost(member_ff)

        # Initialize some important variables
        weights = np.ones(self._num_antennas)   # Start it off at uniform, just in case num_iterations = 0
        found = False
        iteration = 0
        num_couples = 20
        num_candidates = 10
        new_population = np.zeros((self._population + 2 * num_couples, self._num_weights))
        new_ff = np.zeros((self._population + 2 * num_couples, self._sine_space.size))
        new_costs = np.zeros((self._population + 2 * num_couples))
        min_costs = np.zeros(8)
        while not found:
            new_population[:self._population, :] = members
            new_ff[:self._population, :] = member_ff
            new_costs[:self._population] = member_scores
            children = []

            mut_rate, mut_range, cross_rate = self.get_params(iteration)

            for k in range(num_couples):
                # Get the parents (best 2) from a random sampling of the population
                candidates = random.sample(range(self._population), num_candidates)
                idx = np.argmin(member_scores[candidates])
                madre_idx = candidates[idx]
                candidates.pop(idx)
                idx = np.argmin(member_scores[candidates])
                padre_idx = candidates[idx]

                # Make 2 children from those parents
                child1, child2 = self.make_children(members[madre_idx, :], members[padre_idx, :],
                                                    cross_rate, mut_rate, mut_range)
                children += [child1, child2]

            # Score the children
            children_ff = self.calculate_far_field(np.array(children))
            children_scores = self.cost(children_ff)
            new_population[self._population:, :] = children
            new_ff[self._population:, :] = children_ff
            new_costs[self._population:] = children_scores

            # Only keep the best
            sorted_indices = np.argsort(new_costs)
            members = new_population[sorted_indices[:self._population], :]
            member_scores = new_costs[sorted_indices[:self._population]]
            member_ff = new_ff[sorted_indices[:self._population], :]

            self._scores.append(member_scores[0])
            min_costs[iteration % 8] = member_scores[0]

            if iteration % 8 == 7:
                self.update_params(min_costs)
                self._cycle_idx += 1
                if self._cycle_idx >= len(self._param_cycle):
                    self._cycle_idx = 0
                min_costs = np.zeros(8)

            # Found a solution! Exit early.
            if member_scores[0] == 0:
                found = True

            if iteration % 1000 == 0 and iteration != 0:
                if (self._scores[iteration - 1000] - self.best_score) / self.best_score < 1e-4:
                    # Score has stagnated - found a local or global minima
                    found = True

            weights = np.ones(self._num_antennas, dtype=np.complex128)
            weights[1:self._num_weights+1] = np.exp(1j * members[0, :])
            weights[self._num_weights+1:-1] = np.exp(1j * np.flip(members[0, :]))
            iteration += 1
            if found:
                print("Iteration: {}\tCost: {}".format(iteration - 1, member_scores[0]))
                # plt.plot(self._sine_space, self._upper_bound, label='Upper')
                # plt.plot(self._sine_space, self._lower_bound, label='Lower')
                # plt.plot(self._sine_space, member_ff[0, :], label='Example')
                # # plt.ylim([-6, 3])
                # over_upper = np.argwhere(member_ff[0, :] > self._upper_bound)
                # under_lower = np.argwhere(member_ff[0, :] < self._lower_bound)
                # plt.plot(self._sine_space[over_upper], member_ff[0, over_upper], marker='+',
                #          label='Over')
                # plt.plot(self._sine_space[under_lower], member_ff[0, under_lower], marker='+',
                #          label='Under')
                # plt.legend()
                # plt.show()
        return weights

    def update_params(self, costs):
        """Update the parameters and their directions if need be."""
        # Figure out how to adapt the parameters
        rate_idx = self._param_cycle[self._cycle_idx][0]
        range_idx = self._param_cycle[self._cycle_idx][1]
        cross_idx = self._param_cycle[self._cycle_idx][2]

        rate_0_cost = np.min(costs[np.argwhere(self._parameter_orders[:, rate_idx] == 0)])
        rate_1_cost = np.min(costs[np.argwhere(self._parameter_orders[:, rate_idx] == 1)])
        range_0_cost = np.min(costs[np.argwhere(self._parameter_orders[:, range_idx] == 0)])
        range_1_cost = np.min(costs[np.argwhere(self._parameter_orders[:, range_idx] == 1)])
        cross_0_cost = np.min(costs[np.argwhere(self._parameter_orders[:, cross_idx] == 0)])
        cross_1_cost = np.min(costs[np.argwhere(self._parameter_orders[:, cross_idx] == 1)])

        if rate_0_cost < rate_1_cost:
            self._mut_rate_dir *= -1
        else:
            self._mut_rate_idx += self._mut_rate_dir
        if self._mut_rate_idx == self._mutation_rates.size - 1:
            self._mut_rate_dir = -1
        elif self._mut_rate_idx == 0:
            self._mut_rate_dir = 1

        if range_0_cost < range_1_cost:
            self._mut_range_dir *= -1
        else:
            self._mut_range_idx += self._mut_range_dir
        if self._mut_range_idx == self._mutation_ranges.size - 1:
            self._mut_range_dir = -1
        elif self._mut_range_idx == 0:
            self._mut_range_dir = 1

        if cross_0_cost < cross_1_cost:
            self._cross_dir *= -1
        else:
            self._cross_idx += self._cross_dir
        if self._cross_idx == self._crossovers.size - 1:
            self._cross_dir = -1
        elif self._cross_idx == 0:
            self._cross_dir = 1

    def get_params(self, iteration):
        """Get the mutation rate, mutation range, and crossover rate for this iteration."""
        # These will be either 0 or 1 - 0 for current parameter, 1 for comparison parameter
        rate_choice = self._parameter_orders[iteration % 8, self._param_cycle[self._cycle_idx][0]]
        range_choice = self._parameter_orders[iteration % 8, self._param_cycle[self._cycle_idx][1]]
        over_choice = self._parameter_orders[iteration % 8, self._param_cycle[self._cycle_idx][2]]

        mut_rate = self._mutation_rates[self._mut_rate_idx + rate_choice * self._mut_rate_dir]
        mut_range = self._mutation_ranges[self._mut_range_idx + range_choice * self._mut_range_dir]
        cross_rate = self._crossovers[self._cross_idx + over_choice * self._cross_dir]

        return mut_rate, mut_range, cross_rate

    def make_children(self, parent1, parent2, num_crossovers, mutation_rate, mutation_range):
        """Make num_children from the two provided parents."""
        # Determine the number of crossover events.
        min_crossovers = np.floor(num_crossovers)
        max_crossovers = np.ceil(num_crossovers)
        prob = num_crossovers - min_crossovers
        num = np.random.uniform(0, 1, 1)
        if num < prob:
            num_crosses = min_crossovers
        else:
            num_crosses = max_crossovers

        # Determine where the crossovers will occur
        locations = sorted(random.sample(range(self._num_weights), int(num_crosses)))

        # Make the children
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        for i, loc in enumerate(locations):
            if i % 2 == 0:
                child1[loc:] = parent2[loc:]
                child2[loc:] = parent1[loc:]
            else:
                child1[loc:] = parent1[loc:]
                child2[loc:] = parent2[loc:]

        # Mutations
        for kid in [child1, child2]:
            rand_samps = np.random.uniform(0, 1, kid.shape)
            sites = np.argwhere(rand_samps < mutation_rate)
            kid[sites] = kid[sites] + np.random.uniform(-mutation_range/2, mutation_range/2, sites.shape)

        return child1, child2

    def initialize_population(self):
        """Randomly select 'population' sets of weights to initialize the algorithm."""
        return np.random.uniform(0, 1, (self._population, self._num_weights)) * 2 * np.pi

    def calculate_far_field(self, members):
        """Computes the far field pattern for the given members."""
        symmetric_members = np.zeros((members.shape[0], self._num_antennas))
        symmetric_members[:, 1:self._num_weights+1] = members
        symmetric_members[:, self._num_weights+1:-1] = np.fliplr(members)
        ff = self.array_factor(np.exp(1j * symmetric_members))
        scale = 1 / np.max(np.abs(ff), axis=1)    # Normalize so that max amplitude is 1 in each far field pattern
        ff = np.einsum('i,ij->ij', scale, ff)
        return 20 * np.log10(np.abs(ff))

    def array_factor(self, weights):
        """Computes the array factor for the given array configuration.

        Parameters
        ----------
        weights: np.ndarray
            Array of weighting factors for each antenna. Each row is for one configuration

        Returns
        -------
        af: np.ndarray
            Array factor of radar array, at all points in [thetas, phis].
        """
        k = 2 * np.pi * self._freq / c

        spatial_grid = k * self._sine_space
        exponents = np.einsum('i,j->ij', self._antenna_positions, spatial_grid)
        v = np.exp(1j * exponents)

        af = np.einsum('ij,jk->ik', weights, v) / len(self._antenna_positions)

        return af

    def cost(self, ff_pattern):
        """Compute the cost function for a given far-field pattern"""
        costs = np.zeros(ff_pattern.shape[0])
        for i in range(ff_pattern.shape[0]):

            over_upper = np.argwhere(ff_pattern[i, :] > self._upper_bound)
            under_lower = np.argwhere(ff_pattern[i, :] < self._lower_bound)

            if ff_pattern.shape[0] == 1:
                plt.plot(self._sine_space, self._upper_bound, label='Upper')
                plt.plot(self._sine_space, self._lower_bound, label='Lower')
                plt.plot(self._sine_space, ff_pattern[i, :], label='Example')
                plt.plot(self._sine_space[over_upper], ff_pattern[i, over_upper], marker='+', label='Over')
                plt.plot(self._sine_space[under_lower], ff_pattern[i, under_lower], marker='+', label='Under')
                plt.legend()
                plt.show()

            over_cost = np.sum(np.square(ff_pattern[i, over_upper] - self._upper_bound[over_upper]))
            under_cost = np.sum(np.square(self._lower_bound[under_lower] - ff_pattern[i, under_lower]))
            costs[i] = over_cost + under_cost
        return costs

    @property
    def weights(self):
        return self._weights

    @property
    def best_score(self):
        return self._scores[-1]

    @property
    def all_scores(self):
        return self._scores
