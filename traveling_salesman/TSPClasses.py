#!/usr/bin/python3


import math
import numpy as np
import random
import time


class TSPSolution:
	def __init__(self, cities):
		self.route = cities
		self.cost = self.get_route_cost()

	def get_route_cost(self):
		'''
		:return: the total cost of all edges on a particular route
		'''
		cost = 0
		previous_city = self.route[0]
		for current_city in self.route[1:]:
			cost += previous_city.get_cost_btw_cities(current_city)
			previous_city = current_city
		cost += self.route[-1].get_cost_btw_cities(self.route[0])

		return cost

	def get_edges(self):
		'''
		:return: all edges (New York City, Boston, 200 mi) if there's a valid path else None
		'''
		edges = []

		previous_city = self.route[0]
		for current_city in self.route[1:]:
			dist = previous_city.get_cost_btw_cities(current_city)
			if dist == np.inf:
				return None
			edges.append((previous_city, current_city, int(math.ceil(dist))))
			previous_city = current_city
		dist = self.route[-1].get_cost_btw_cities(self.route[0])
		if dist == np.inf:
			return None
		edges.append((self.route[-1], self.route[0], int(math.ceil(dist))))

		return edges


def get_name_for_int(num):
	'''
	:param num: a whole number (City number)
	:return: A, B, ..., Z, AA, BB, etc., depending on num
	'''
	if num == 0:
		return ''
	elif num <= 26:
		return chr(ord('A') + num - 1)
	else:
		return get_name_for_int((num-1) // 26) + get_name_for_int((num-1) % 26 + 1)


class Scenario:
	def __init__(self, city_coordinates, difficulty, random_seed):
		self.difficulty = difficulty
		self.HARD_MODE_FRACTION_TO_REMOVE = 0.20  # Remove 20% of the edges

		if difficulty == "Normal" or difficulty == "Hard":
			self.cities = [City(coordinate.x(), coordinate.y(), random.uniform(0.0, 1.0)) for coordinate in city_coordinates]
		elif difficulty == "Hard (Deterministic)":
			random.seed(random_seed)
			self.cities = [City(coordinate.x(), coordinate.y(), random.uniform(0.0, 1.0)) for coordinate in city_coordinates]
		else:
			self.cities = [City(coordinate.x(), coordinate.y()) for coordinate in city_coordinates]

		for i, city in enumerate(self.cities):
			# if difficulty == "Hard":
			city.set_scenario(self)
			city.set_index_and_name(i, get_name_for_int(i + 1))

		# Assume all edges exists except self-edges
		num_cities = len(self.cities)
		self.edge_exists = (np.ones((num_cities, num_cities)) - np.diag(np.ones(num_cities))) > 0

		if difficulty == "Hard":
			self.remove_edges()
		elif difficulty == "Hard (Deterministic)":
			self.remove_edges(deterministic=True)

	def get_cities(self):
		return self.cities

	def scramble_order(self, n):
		'''
		:param n: an integer
		:return: a size-n numpy array of scrambled order
		'''
		permutation = np.arange(n)
		for i in range(n):
			random_integer = random.randint(i, n - 1)

			tmp = permutation[i]
			permutation[i] = permutation[random_integer]
			permutation[random_integer] = tmp

		return permutation

	def remove_edges(self, deterministic=False):
		'''
		remove edges by setting their corresponding bools in the NumPy 2-d array to False
		'''
		num_cities = len(self.cities)
		num_edges = num_cities * (num_cities - 1)  # can't have self-edge
		num_edges_to_remove = np.floor(self.HARD_MODE_FRACTION_TO_REMOVE * num_edges)

		can_delete = self.edge_exists.copy()

		# Set aside a route to ensure at least one tour exists
		route_to_keep = np.random.permutation(num_cities)
		if deterministic:
			route_to_keep = self.scramble_order(num_cities)
		for i in range(num_cities):
			can_delete[route_to_keep[i], route_to_keep[(i + 1) % num_cities]] = False

		# Now remove edges until 
		while num_edges_to_remove > 0:
			if deterministic:
				src = random.randint(0, num_cities - 1)
				dst = random.randint(0, num_cities - 1)
			else:
				src = np.random.randint(num_cities)
				dst = np.random.randint(num_cities)
			if self.edge_exists[src, dst] and can_delete[src, dst]:
				self.edge_exists[src, dst] = False
				num_edges_to_remove -= 1


class City:
	def __init__(self, x, y, elevation=0.0):
		self.x = x
		self.y = y
		self.elevation = elevation
		self.scenario = None
		self.index = -1
		self.name = None
		self.MAP_SCALE = 1000.0

	def set_index_and_name(self, index, name):
		self.index = index
		self.name = name

	def set_scenario(self, scenario):
		self.scenario = scenario

	def get_cost_btw_cities(self, destination):
		'''
		:param destination: the adjacent city we wanna travel to
		:return: the cost between the current city and the adjacent city
		'''
		assert(type(destination) == City)

		# In hard mode, remove edges; this slows down the calculation...
		# Use this in all difficulties, it ensures INF for self-edge
		if not self.scenario.edge_exists[self.index, destination.index]:
			return np.inf

		# Euclidean Distance
		cost = math.sqrt((destination.x - self.x)**2 + (destination.y - self.y)**2)

		# For Medium and Hard modes, add in an asymmetric cost (in easy mode it is zero).
		if not self.scenario.difficulty == 'Easy':
			cost += (destination.elevation - self.elevation)
			if cost < 0.0:
				cost = 0.0

		return int(math.ceil(cost * self.MAP_SCALE))