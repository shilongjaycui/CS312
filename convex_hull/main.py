from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF, QObject
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF, QObject
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))



import time
import math

# Some global color constants that might be useful
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

# Global variable that controls the speed of the recursion automation, in seconds
#
PAUSE = 0.25

#
# This is the class you have to complete.
#
class ConvexHullSolver(QObject):

# Class constructor
	def __init__( self):
		super().__init__()
		self.pause = False
		
# Some helper methods that make calls to the GUI, allowing us to send updates
# to be displayed.
	def showTangent(self, line, color):
		self.view.addLines(line,color)
		if self.pause:
			time.sleep(PAUSE)

	def eraseTangent(self, line):
		self.view.clearLines(line)

	def blinkTangent(self,line,color):
		self.showTangent(line,color)
		self.eraseTangent(line)

	def showHull(self, polygon, color):
		self.view.addLines(polygon,color)
		if self.pause:
			time.sleep(PAUSE)
		
	def eraseHull(self,polygon):
		self.view.clearLines(polygon)
		
	def showText(self,text):
		self.view.displayStatusText(text)

# Other helper functions
	def get_sorted_points(self, points, orientation):
		'''
		points: a list of points (QPointF objects);
		orientation: 'clockwise' or 'counterclockwise'
		returns the list of points sorted
		Time: O(n log n)
		Space: O(n)
		'''
		center = QPointF(sum([point.x() for point in points]) / len(points), sum([point.y() for point in points]) / len(points))	# time: O(1); space: O(n)

		def normalize(p, c):
			return p.x() - c.x(), p.y() - c.y()	# time & space: O(1)

		def get_radian(p):
			return math.atan2(p[1], p[0])	# time & space: O(1)

		if orientation == 'counterclockwise':
			return sorted(points, key=lambda point: get_radian(normalize(point, center)))	# time: O(n); space: O(n)
		elif orientation == 'clockwise':
			return sorted(points, key=lambda point: get_radian(normalize(point, center)), reverse=True)	# time: O(n); space: O(n)

	def get_slope(self, p1, p2):
		'''
		p1, p2: points (QPointF objects)
		Time: O(1)
		Space: O(1)
		'''
		return (p2.y() - p1.y()) / (p2.x() - p1.x())

	def is_tangent(self, left, right, i, j, upper_or_lower, left_or_right):
		'''
		left: the list of left hull points;
		right: the list of right hull points;
		i: the index of the left hull point;
		j: the index of the right hull point;
		upper_or_lower: 'upper' or 'lower';
		left_or_right: 'left' or 'right';
		returns whether the line passing through left[i] and right[j] is upper/lower left/right tangent or not
		Time: O(1)
		Space: O(1)
		'''
		# if i >= len(left):
		# 	i = i % len(left)
		# if j >= len(right):
		# 	j = j % len(right)

		slope = self.get_slope(left[i], right[j])	# time & space: O(1)

		if left_or_right == 'left':
			new_slope = self.get_slope(left[(i+1) % len(left)], right[j])	# time & space: O(1) (we're treating minor operations like + and % as constant time)
			if upper_or_lower == 'upper':
				if new_slope > slope:
					return True
			elif upper_or_lower == 'lower':
				if new_slope < slope:
					return True
		elif left_or_right == 'right':
			new_slope = self.get_slope(left[i], right[(j+1) % len(right)])	# time & space: O(1)
			if upper_or_lower == 'upper':
				if new_slope < slope:
					return True
			elif upper_or_lower == 'lower':
				if new_slope > slope:
					return True

		return False

	def compute_hull_helper(self, points):
		'''
		points: a list of points (QPointF objects) sorted by ascending x-value;
		returns the list of hull points
		Time: O(n log n)
		Space: O(n)
		'''
		if len(points) < 4:
			return points

		mid = len(points) // 2	# time & space: O(1)
		left_points, right_points = points[:mid], points[mid:]	# time: O(1); space: O(n)
		return self.merge(self.compute_hull_helper(left_points), self.compute_hull_helper(right_points))	# time: O(n log n); space: O(log n)

	def merge(self, left, right):
		'''
		left: the list of left hull points;
		right: the list of right hull points;
		returns the list of hull points
		Time: O(n log n)
		Space: O(n)
		'''
		points = []	# time: O(1); space: O(n)

		left_counterclockwise = self.get_sorted_points(left, 'counterclockwise')	# time: O(n); space: O(n)
		right_clockwise = self.get_sorted_points(right, 'clockwise')	# time: O(n); space: O(n)

		i = left_counterclockwise.index(max(left_counterclockwise, key=lambda c: c.x()))	# time: O(n); space: O(1)
		j = right_clockwise.index(min(right_clockwise, key=lambda c: c.x()))	# time: O(n); space: O(1)
		while not self.is_tangent(left_counterclockwise, right_clockwise, i, j, 'upper', 'left') or not self.is_tangent(left_counterclockwise, right_clockwise, i, j, 'upper', 'right'):	# time: O(n); space: O(1)
			while not self.is_tangent(left_counterclockwise, right_clockwise, i, j, 'upper', 'left'):
				i = (i + 1) % len(left_counterclockwise)
			while not self.is_tangent(left_counterclockwise, right_clockwise, i, j, 'upper', 'right'):
				j = (j + 1) % len(right_clockwise)

		point_upper_left = left_counterclockwise[i]
		points.append(point_upper_left)
		point_upper_right = right_clockwise[j]
		points.append(point_upper_right)

		left_clockwise = left_counterclockwise[::-1]	# time: O(n); space: O(n)
		right_counterclockwise = right_clockwise[::-1]

		i = left_clockwise.index(max(left_clockwise, key=lambda c: c.x()))
		j = right_counterclockwise.index(min(right_counterclockwise, key=lambda c: c.x()))
		while not self.is_tangent(left_clockwise, right_counterclockwise, i, j, 'lower', 'left') or not self.is_tangent(left_clockwise, right_counterclockwise, i, j, 'lower', 'right'):
			while not self.is_tangent(left_clockwise, right_counterclockwise, i, j, 'lower', 'left'):
				i = (i + 1) % len(left_clockwise)
			while not self.is_tangent(left_clockwise, right_counterclockwise, i, j, 'lower', 'right'):
				j = (j + 1) % len(right_counterclockwise)

		point_lower_left = left_clockwise[i]
		points.append(point_lower_left)
		point_lower_right = right_counterclockwise[j]
		points.append(point_lower_right)

		i = left_counterclockwise.index(point_upper_left)
		j = left_counterclockwise.index(point_lower_left)
		k = (i + 1) % len(left_counterclockwise)
		while k != j:	# time: O(n); space: O(1)
			points.append(left_counterclockwise[k])
			k = (k + 1) % len(left_counterclockwise)

		i = right_clockwise.index(point_upper_right)
		j = right_clockwise.index(point_lower_right)
		k = (i + 1) % len(right_clockwise)
		while k != j:	# time: O(n); space: O(1)
			points.append(right_clockwise[k])
			k = (k + 1) % len(right_clockwise)

		return points



# This is the method that gets called by the GUI and actually executes
# the finding of the hull
	def compute_hull( self, points, pause, view):
		'''
		Time: O(n log n)
		Space: O(log n)
		'''
		self.pause = pause
		self.view = view
		assert( type(points) == list and type(points[0]) == QPointF )

		t1 = time.time()
		# TODO: SORT THE POINTS BY INCREASING X-VALUE

		points.sort(key=lambda coordinate: coordinate.x())	# Timsort: O(n log n)
		# print('Points sorted by x-values:', self.QPointFtoTuple(points), '\n')

		t2 = time.time()
		start = time.process_time()

		polygon_points = self.get_sorted_points(self.compute_hull_helper(points), 'counterclockwise')	# time: O(n log n); space: O(log n)

		t3 = time.time()
		print('Time elapsed (s):', time.process_time() - start)
		
		polygon = [QLineF(polygon_points[i], polygon_points[(i+1) % len(polygon_points)]) for i in range(len(polygon_points))]	# POLYGON POINTS NEED TO BE SORTED CLOCKWISE/COUNTERCLOCKWISE IN ORDER TO DO THIS

		# TODO: REPLACE THE LINE ABOVE WITH A CALL TO YOUR DIVIDE-AND-CONQUER CONVEX HULL SOLVER
		t4 = time.time()

		# when passing lines to the display, pass a list of QLineF objects.  Each QLineF
		# object can be created with two QPointF objects corresponding to the endpoints
		self.showHull(polygon,RED)
		self.showText('Time Elapsed (Convex Hull): {:3.3f} sec'.format(t4-t3))



