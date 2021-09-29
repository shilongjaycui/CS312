#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


class Cell:
	def __init__(self, coordinate, cost, row_base, col_base, prev=None):
		'''
		:param coordinate: (i, j), where i is the cell's row index and j is the cell's column index
		:param cost: the cost of computing this particular sub-alignment
		:param row_base: the i-th base in seq1 (the vertical sequence)
		:param col_base: the j-th base in seq2 (the horizontal sequence)
		:param prev: the left/top/diagonal cell that ensures the minimum cost of this cell
		'''
		self.coordinate = coordinate
		self.row_base = row_base
		self.col_base = col_base
		self.cost = cost
		self.prev = prev


class GeneSequencing:
	def __init__(self):
		self.substitution_cost = 1
		self.insertion_cost = 5
		self.deletion_cost = 5
		self.match_reward = -3
		self.d = 3

	def update_cost(self, matrix, i, j, banded):
		'''
		Updates the cost of the cell at row i and column j
		:param matrix: the cost matrix
		:param i: row index
		:param j: column index
		:param banded: whether we're using the banded algorithm or not
		:return: None
		Time: O(1)
		Space: O(1)
		'''
		# find the current cell and its neighbors
		if banded:
			def find_banded_cell(table, row_index, col_index):
				'''
				:param table: the banded matrix
				:param row_index: i
				:param col_index: j
				:return: the cell with coordinates (i, j)
				Time: O(1)
				Space: O(1)
				'''
				row = table[row_index]
				for entry in row:
					if entry and entry.coordinate[1] == col_index:
						return entry
					else:
						continue
				return None

			cell = find_banded_cell(matrix, i, j)
			left = find_banded_cell(matrix, i, j-1)
			top = find_banded_cell(matrix, i-1, j)
			diagonal = find_banded_cell(matrix, i-1, j-1)
		else:
			cell = matrix[i][j]
			left = matrix[i][j - 1]
			top = matrix[i - 1][j]
			diagonal = matrix[i - 1][j - 1]

		# find the costs of the cell's neighbors
		if left:
			left_cost = left.cost + self.insertion_cost
		else:
			left_cost = float('inf')
		if top:
			top_cost = top.cost + self.deletion_cost
		else:
			top_cost = float('inf')
		if cell.col_base == cell.row_base:
			diagonal_cost = diagonal.cost + self.match_reward
		else:
			diagonal_cost = diagonal.cost + self.substitution_cost

		# update the cell's cost and previous cell
		cell.cost = left_cost
		cell.prev = left
		if top_cost < cell.cost:
			cell.cost = top_cost
			cell.prev = top
		if diagonal_cost < cell.cost:
			cell.cost = diagonal_cost
			cell.prev = diagonal

	def create_matrix(self, seq1, seq2):
		'''
		:param seq1: 1st gene sequence
		:param seq2: 2nd gene sequence
		:return: the full cost matrix of seq1 and seq2
		Time: O(nm)
		Space: O(nm)
		'''
		# dummy table records and headers to help create the matrix
		row_bases = '-' + seq1
		col_bases = '-' + seq2
		# initialize the matrix
		matrix = [[Cell((i, j), 0, row_bases[i], col_bases[j]) for j in range(len(col_bases))] for i in range(len(row_bases))]	# O(nm)
		self.mat_num_row = len(matrix)
		self.mat_num_col = len(matrix[0])

		# populate the first row and the first column with the default insertion/deletion costs
		num_delete = 0
		while num_delete < self.mat_num_col:	# O(m)
			matrix[0][num_delete].cost += num_delete * self.deletion_cost
			num_delete += 1

		num_insert = 0
		while num_insert < self.mat_num_row:	# O(n)
			matrix[num_insert][0].cost += num_insert * self.insertion_cost
			num_insert += 1

		# iterate the sub-matrix (the original matrix without the first row or the first column)
		for i in range(1, self.mat_num_row):	# O(n)
			for j in range(1, self.mat_num_col):	# O(m)
				# update the cell's cost and previous cell
				self.update_cost(matrix, i, j, False)	# O(1)

		return matrix

	def create_banded_matrix(self, seq1, seq2):
		'''
		:param seq1: 1st gene sequence
		:param seq2: 2nd gene sequence
		:return: the banded cost matrix of seq1 and seq2
		Time: O(kn)
		Space: O(kn)
		'''
		row_bases = '-' + seq1
		col_bases = '-' + seq2
		matrix = []
		for i in range(len(row_bases)):	# O(n)
			row = []
			for j in range(i - self.d, i + self.d + 1):	# O(k)
				if j < 0:
					row.append(None)
				elif j >= len(col_bases):
					row.append(None)
				else:
					row.append(Cell((i, j), float('inf'), row_bases[i], col_bases[j]))
			matrix.append(row)
		self.mat_num_row = len(matrix)
		self.mat_num_col = len(matrix[0])

		for k in range(4):	# O(1)
			matrix[0][k+3].cost = k * self.deletion_cost
			matrix[k][3-k].cost = k * self.insertion_cost

		for i in range(1, len(row_bases)):	# O(n)
			for j in range(max(1, i - self.d), min(i + self.d + 1, len(col_bases))):	# O(k)
				self.update_cost(matrix, i, j, True)

		return matrix

	def find_last_cell(self, matrix):
		'''
		:param matrix: the cost matrix
		:return: the cell at matrix's bottom right corner
		Time: O(1)
		Space: O(1)
		'''
		return [cell for cell in matrix[-1] if cell is not None][-1]

	def extract_alignment(self, matrix):
		'''
		:param matrix: the cost matrix
		:return: 1st gene sequence's alignment, 2nd gene sequence's alignment
		Time (unrestricted): O(nm)
		Space (unrestricted): O(nm)
		Time (banded): O(kn)
		Space (banded): O(kn)
		'''
		# find the bottom right corner cell of the matrix
		cell = self.find_last_cell(matrix)	# O(1)

		# initialize the reversed base stacks
		seq1_stack = []
		seq2_stack = []

		# create helper functions that identify operation (insert/delete/substitution/match)
		def is_insertion(current_cell, previous_cell):
			'''
			:return: whether we did an insertion to get from previous_cell to current_cell
			Time: O(1)
			Space: O(1)
			'''
			return previous_cell.coordinate[0] == current_cell.coordinate[0] and previous_cell.coordinate[1] == current_cell.coordinate[1] - 1

		def is_deletion(current_cell, previous_cell):
			'''
			:return: whether we did a deletion to get from previous_cell to current_cell
			Time: O(1)
			Space: O(1)
			'''
			return previous_cell.coordinate[0] == current_cell.coordinate[0] - 1 and previous_cell.coordinate[1] == current_cell.coordinate[1]

		# populate the stacks with bases found along the back pointer path
		while cell.prev:	# O(nm) (unrestricted); O(kn) (banded)
			# check what operation it is (insert/delete/substitution/match)
			if is_insertion(cell, cell.prev):
				seq1_stack.append('-')
				seq2_stack.append(cell.col_base)
			elif is_deletion(cell, cell.prev):
				seq1_stack.append(cell.row_base)
				seq2_stack.append('-')
			else:
				seq1_stack.append(cell.row_base)
				seq2_stack.append(cell.col_base)

			cell = cell.prev

		# create alignments by popping the stacks
		seq1_alignment = []
		while seq1_stack:	# O(nm) (unrestricted); O(kn) (banded)
			seq1_alignment.append(seq1_stack.pop())
		seq2_alignment = []
		while seq2_stack:	# O(nm) (unrestricted); O(kn) (banded)
			seq2_alignment.append(seq2_stack.pop())

		return "".join(seq1_alignment), "".join(seq2_alignment)

	def align(self, seq1, seq2, banded, align_length):
		'''
		:param seq1: 1st sequence to be aligned
		:param seq2: 2nd sequence to be aligned
		:param banded: whether you should compute a banded alignment or full alignment (bool)
		:param align_length: how many base pairs to use in computing the alignment
		:return: alignment cost, 1st gene sequence alignment, 2nd gene sequence alignment
		Time (unrestricted): O(nm)
		Space (unrestricted): O(nm)
		Time (banded): O(kn)
		Space (banded): O(kn)
		'''
		self.banded = banded
		self.MaxCharactersToAlign = align_length

		# get rid of newline characters in the sequences
		seq1 = seq1.replace('\n', '')
		seq2 = seq2.replace('\n', '')

		# align the first n characters (bases) in each sequence pair
		seq1 = seq1[:self.MaxCharactersToAlign]
		seq2 = seq2[:self.MaxCharactersToAlign]

		# make sure the first sequence is longer than or of equal length as the second sequence (might be redundant)
		if len(seq1) < len(seq2):
			seq1, seq2 = seq2, seq1

		if banded:
			if len(seq1) - len(seq2) > 3:
				return {'align_cost': float('inf'), 'seqi_first100': 'No Alignment Possible', 'seqj_first100': 'No Alignment Possible'}
			matrix = self.create_banded_matrix(seq1, seq2)	# O(kn)
			score = self.find_last_cell(matrix).cost	# O(1)
		else:
			matrix = self.create_matrix(seq1, seq2)	# O(nm)
			score = matrix[self.mat_num_row - 1][self.mat_num_col - 1].cost

		alignment1 = self.extract_alignment(matrix)[0]
		alignment2 = self.extract_alignment(matrix)[1]

		return {'align_cost': score, 'seqi_first100': alignment1, 'seqj_first100': alignment2}

'''
archived: create_banded_matrix()
		# row_bases = '-' + seq1
		# col_bases = '-' + seq2
		#
		# matrix = [[Cell((i, j), float('inf'), row_bases[i], col_bases[j]) for j in range(len(col_bases))] for i in range(len(row_bases))]
		#
		# self.mat_num_row = len(matrix)
		# self.mat_num_col = len(matrix[0])
		#
		# # create a helper function that tells whether an alignment is within the specified bandwidth
		# def within_bandwidth(i, j):
		# 	return abs(i - j) <= self.d
		#
		# num_delete = 0
		# while num_delete < self.mat_num_col:
		# 	if within_bandwidth(0, num_delete):
		# 		matrix[0][num_delete].cost = num_delete * self.deletion_cost
		# 		num_delete += 1
		# 	else:
		# 		break
		#
		# num_insert = 0
		# while num_insert < self.mat_num_row:
		# 	if within_bandwidth(num_insert, 0):
		# 		matrix[num_insert][0].cost = num_insert * self.insertion_cost
		# 		num_insert += 1
		# 	else:
		# 		break
		#
		# for i in range(1, self.mat_num_row):
		# 	starting_column_index = max(1, i - self.d)
		# 	for j in range(starting_column_index, self.mat_num_col):
		# 		if within_bandwidth(i, j):
		# 			self.update_cost(matrix, i, j)
		# 		else:
		# 			break
		#
		# return matrix
'''