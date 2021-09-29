#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import copy


class State:  # data structure to represent the states
	def __init__(self, partial_path, city, city_indices, cost_matrix, lower_bound):
		'''
		Time: O(n)
		Space: O(n^2)
		'''
		# cities on the path
		self.partial_path = copy.deepcopy(partial_path)  # space: O(n)
		self.partial_path.append(city)

		# indices of cities on the path
		self.city_indices = copy.deepcopy(city_indices)  # space: O(n)
		self.city_indices.add(city.index)

		self.cost_matrix = cost_matrix  # space: O(n^2)

		self.lower_bound = lower_bound

		# Reduce the cost matrix
		self.reduce_cost_matrix()  # time: O(n); space: O(n)

	def __lt__(self, other):
		return self.lower_bound < other.lower_bound

	def reduce_cost_matrix(self):
		'''
		Time: O(n)
		Space: O(n)
		'''
		row_min = np.amin(self.cost_matrix, axis=1)  # space: O(n)
		row_min = np.where(row_min == float('inf'), 0, row_min)

		assert(row_min.size == len(self.cost_matrix))

		for j in range(len(self.cost_matrix)):  # time: O(n)
			self.cost_matrix[j] = self.cost_matrix[j] - row_min[j]
			self.lower_bound += row_min[j]

		col_min = np.amin(self.cost_matrix, axis=0)  # space: O(n)
		col_min = np.where(col_min == float('inf'), 0, col_min)

		assert(col_min.size == len(self.cost_matrix))

		for j in range(len(self.cost_matrix)):  # time: O(n)
			self.cost_matrix[:, j] = self.cost_matrix[:, j] - col_min[j]
			self.lower_bound += col_min[j]

	def get_substates(self, cities):
		'''
		Time: O(n^2)
		Space: O(n^2)
		'''
		i = self.partial_path[-1].index  # starting city index

		substates = []
		for j in range(len(cities)):  # time: O(n)
			if j not in self.city_indices:
				city = cities[j]
				cost_matrix, lower_bound = self.get_updated_matrix(i, j)  # space: O(n^2)
				substates.append(State(self.partial_path, city, self.city_indices, \
									   cost_matrix, lower_bound))  # time: O(n), space: O(n^2)

		assert(len(substates) == len(cities) - len(self.partial_path))

		return substates

	def get_updated_matrix(self, i, j):
		'''
		Time: O(1)
		Space: O(n^2)
		'''
		cost_matrix = copy.deepcopy(self.cost_matrix)  # space: O(n^2)
		lower_bound = copy.deepcopy(self.lower_bound)

		# increment LB with the cost at (i, j)
		lower_bound += cost_matrix[i][j]
		# set the i-th row, j-th column, and (j, i) to infinity
		cost_matrix[i] = cost_matrix[i] + float('inf')
		cost_matrix[:, j] = cost_matrix[:, j] + float('inf')
		cost_matrix[j][i] = float('inf')

		return cost_matrix, lower_bound

	def is_solution(self):
		return len(self.partial_path) == len(self.cost_matrix)


class TSPSolver:
	def __init__(self, gui_view):
		self.scenario = None

	def set_scenario(self, scenario):
		self.scenario = scenario

	# ----------------------------------------------------------------------------------------------------

	# helper function for greedy
	def get_neighbors_costs(self, city, cities, edge_exists, cost_matrix):
		'''
		Time: O(n)
		Space: O(n)
		'''
		neighbors_costs = []  # space: O(n)

		edges = edge_exists[city.index]
		costs = cost_matrix[city.index]
		assert(len(edges) == len(costs))
		for j in range(len(edges)):  # time: O(n)
			if edges[j]:
				neighbors_costs.append((cities[j], costs[j]))

		return neighbors_costs

	# helper function for branch and bound
	def generate_key(self, state):
		'''
		Time: O(1)
		Space: O(1)
		'''
		lower_bound = state.lower_bound  # want this to be SMALL
		tree_depth = len(state.partial_path)  # want this to be BIG

		return lower_bound - 500 * tree_depth

	# ----------------------------------------------------------------------------------------------------

	def default(self, time_allowance=60.0):
		results = {}
		cities = self.scenario.get_cities()
		num_cities = len(cities)
		found = False
		count = 0
		BSSF = None
		start_time = time.time()
		while not found and time.time() - start_time < time_allowance:
			# create a random permutation
			permutation = np.random.permutation(num_cities)
			route = []

			# Now build the route using the random permutation
			for i in range(num_cities):
				route.append(cities[permutation[i]])

			BSSF = TSPSolution(route)

			count += 1
			if BSSF.cost < np.inf:
				# Found a valid route
				found = True

		end_time = time.time()

		results['cost'] = BSSF.cost if found else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = BSSF
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		return results

	def greedy(self, time_allowance=60.0):
		'''
		results dictionary:
		- cost of best solution (int)
		- time spend to find best solution (int)
		- total number of solutions found (int)
		- the best solution found (TSPSolution)
		- max queue size: None
		- total number of states created: None
		- number of pruned states: None

		Time: O(n^3)
		Space: O(n^3)
		'''
		cities = self.scenario.get_cities()  # a Python list of City objects
		num_cities = len(cities)
		start_time = time.time()

		cost_matrix = np.full((num_cities, num_cities), float('inf'))  # space: O(n^2)
		edge_exists = self.scenario.edge_exists  # space: O(n^2)
		assert(cost_matrix.shape == edge_exists.shape)
		for i in range(num_cities):  # time: O(n)
			for j in range(num_cities):  # time: O(n)
				if edge_exists[i][j]:
					cost_matrix[i][j] = cities[i].get_cost_btw_cities(cities[j])

		best_route = {city: float('inf') for city in cities}  # time: O(n)
		best_cost = sum(best_route.values())
		num_solutions = 0
		for starting_city in cities:  # time: O(n)
			route = {starting_city: 0}
			current_city = starting_city
			while len(route) < num_cities:  # time: O(n)
				neighbors_costs = [neighbor_cost for neighbor_cost in \
								   self.get_neighbors_costs(current_city, cities, edge_exists, cost_matrix) \
								   if neighbor_cost[0] not in route]  # time: O(n), space: O(n)
				if len(neighbors_costs) == 0:
					break
				neighbor_cost = min(neighbors_costs, key=lambda neighbor_cost: neighbor_cost[1])
				neighbor = neighbor_cost[0]
				cost = neighbor_cost[1]
				route[neighbor] = cost

				current_city = neighbor

			if len(route) == num_cities:
				homecoming_cost = cost_matrix[current_city.index][starting_city.index]
				if homecoming_cost == float('inf'):
					continue
				else:
					route[starting_city] = homecoming_cost
					cost = sum(route.values())
					if cost < best_cost:
						best_route = route
						best_cost = cost
						num_solutions += 1

		end_time = time.time()

		results = {
			'cost': best_cost,
			'time': end_time - start_time,
			'count': num_solutions,
			'soln': TSPSolution(list(best_route.keys())),
			'max': None,
			'total': None,
			'pruned': None
		}
		return results

	def branch_and_bound(self, time_allowance=60.0):
		'''
		results dictionary:
		- cost of best solution (int)
		- time spent to find best solution (int)
		- total number of solutions found during search (does not include the initial BSSF, int)
		- the best solution found (TSPSolution)
		- max queue size (int)
		- total number of states created (int)
		- number of pruned states (int)

		Time: O(n^2 * (n + 1)!)
		Space: O(n^2 * (n + 1)!)
		'''
		cities = self.scenario.get_cities()
		num_cities = len(cities)
		start_time = time.time()

		# Start at City 1
		cost_matrix = np.full((num_cities, num_cities), float('inf'))  # space: O(n^2)
		edge_exists = self.scenario.edge_exists
		assert(cost_matrix.shape == edge_exists.shape)
		for i in range(num_cities):  # time: O(n)
			for j in range(num_cities):  # time: O(n)
				if edge_exists[i][j]:
					cost_matrix[i][j] = cities[i].get_cost_btw_cities(cities[j])

		state = State([], cities[0], set(), cost_matrix, 0)  # time: O(n); space: O(n^2)

		# Put City 1 in a priority queue
		key = self.generate_key(state)
		queue = [(key, state)]
		heapq.heapify(queue)  # convert the list to a priority queue
		max_queue_size = len(queue)

		# Initialize best solution so far (BSSF)
		BSSF = self.greedy(time_allowance)['soln']  # time: O(n^3); space: O(n^3)

		num_solutions = 0
		num_substates = 0
		num_pruned_states = 0
		while len(queue) > 0 and time.time() - start_time < time_allowance:  # time: O((n + 1)!); space: O((n + 1)!)
			state = heapq.heappop(queue)[1]
			if state.lower_bound >= BSSF.cost:
				num_pruned_states += 1
				continue
			substates = state.get_substates(cities)  # time: O(n^2); space: O(n^2)
			num_substates += len(substates)
			for substate in substates:  # time: O(n)
				if substate.is_solution():
					S = TSPSolution(substate.partial_path)
					if S.cost < BSSF.cost:
						BSSF = S
						num_solutions += 1
				elif substate.lower_bound < BSSF.cost:
					key = self.generate_key(substate)
					heapq.heappush(queue, (key, substate))
				else:
					num_pruned_states += 1

			max_queue_size = max(len(queue), max_queue_size)

		end_time = time.time()

		results = {
			'cost': BSSF.cost,
			'time': end_time - start_time,
			'count': num_solutions,
			'soln': BSSF,
			'max': max_queue_size,
			'total': num_substates,
			'pruned': num_pruned_states
		}
		return results

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
	def fancy(self, time_allowance=60.0):
		start_time = time.time()
		# Step 1: Use greedy to get initial BSSF
		BSSF = self.greedy(time_allowance)['soln']  # BSSF.route: a Python list of City objects; BSSF cost: int

		# Step 2: Go through every pair of edges and swap (i.e. remove one and replace another)
		# n cities -> n edges -> n(n + 1) / 2
		num_cities = len(BSSF.route)
		improved = True
		num_solutions = 1
		while improved and time.time() - start_time < time_allowance:
			improved = False
			# Things that happen when we swap two edges
			# 1. decrement the edge that got swapped out
			for i in range(num_cities):
				for j in range(i + 1, num_cities):
					route = BSSF.route.copy()
					route[i], route[j] = route[j], route[i]
					candidate = TSPSolution(route)
					if candidate.cost < BSSF.cost:
						BSSF = candidate
						improved = True
						num_solutions += 1
				# 		break
				#
				# if improved:
				# 	break

		end_time = time.time()

		results = {
			'cost': BSSF.cost,
			'time': end_time - start_time,
			'count': num_solutions,
			'soln': BSSF,
			'max': None,
			'total': None,
			'pruned': None
		}

		return results