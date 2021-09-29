#!/usr/bin/python3


from CS312Graph import *
import time


class ArrayPQ(list):    # Priority Queue (array implementation)
    def __init__(self):
        list.__init__(self)     # inherit from Python's list data structure

    def insertNode(self, node):
        '''
        Time complexity: O(1)
        Space complexity: O(|V|)
        '''
        self.append(node)

    def deleteMin(self, dist):
        '''
        Time complexity: O(|V|)
        Space complexity: O(|V|)
        '''
        i = 0               # index of the node with the minimum distance in the queue
        m = float('inf')    # minimum distance
        for node in self:
            d = dist[node.node_id]      # find the current node's distance from the starting node
            if d < m:
                m = d
                i = self.index(node)    # index of node in the priority queue

        return self.pop(i)  # O(1)


# from PyQt5.QtCore import QPointF
# import random

# queue = ArrayPQ()
# dists = random.sample(range(1, 11), 5)
# print('\ndists:', dists)
# for i in range(5):
#     node = CS312GraphNode(i, QPointF(i, i))
#     queue.append(node)
#
# print('node_id:', [node.node_id for node in queue])
# print('\nPriority Queue (deleteMin):')
# print('deleted node id:', queue.deleteMin(dists).node_id)
# print('node_id:', [node.node_id for node in queue])


class HeapPQ(list):     # Priority Queue (heap implementation)
    def __init__(self):
        list.__init__(self)     # inherit from Python's list data structure

    def insertNode(self, node, dist):
        '''
        Time complexity: O(log|V|)
        Space complexity: O(|V|)
        '''
        self.append(node)   # O(1)
        self.heapify(self.index(node), dist)    # O(log|V|)

    def heapify(self, index, dist):
        '''
        :param index: Index of the node (0-based) in the priority queue to heapify with respect to
        :param dist: List of graph nodes' distances to the starting node
        :return: None (elements in the priority queue swapped wherever needed)

        Time complexity: O(log|V|) (the depth of the binary heap is log|V|)
        Space complexity: O(|V|)
        '''

        # get heap indices for the node, its parent, and its children
        node_index = index + 1                  # index of the node (1-based): j
        parent_index = node_index // 2          # index of the parent node: j // 2
        left_child_index = 2 * node_index       # index of the left child node: 2j
        right_child_index = 2 * node_index + 1  # index of the right child node: 2j + 1

        # get the node we just inserted/modified
        node = self[node_index - 1]

        if parent_index >= 1:                   # if the node has a parent...
            parent = self[parent_index - 1]     # then we get the parent node
            if not self.isValidHeap(parent, node, dist):
                self[parent_index - 1], self[node_index - 1] = node, parent     # swap
                self.heapify(parent_index - 1, dist)    # recursive call

        if left_child_index <= len(self):               # if the node has a left child...
            left_child = self[left_child_index - 1]     # ...then we get the left child node
            if right_child_index <= len(self):              # if the node as a right child...
                right_child = self[right_child_index - 1]   # ...then we get the right child node
                if not self.isValidHeap(node, left_child, dist) or not self.isValidHeap(node, right_child, dist):
                    if dist[left_child.node_id] <= dist[right_child.node_id]:
                        self[node_index - 1], self[left_child_index - 1] = left_child, node
                        self.heapify(left_child_index - 1, dist)
                    else:
                        self[node_index - 1], self[right_child_index - 1] = right_child, node
                        self.heapify(right_child_index - 1, dist)
            else:   # the node only has a left child, so we compare the node only to its left child
                if not self.isValidHeap(node, left_child, dist):
                    self[node_index - 1], self[left_child_index - 1] = left_child, node
                    self.heapify(left_child_index - 1, dist)

    def isValidHeap(self, parent_node, child_node, dist):
        '''
        :param parent_node: Parent node
        :param child_node: Child node
        :param dist: List of graph nodes' distances to the starting node
        :return: Whether the heap is a valid min-heap or not

        Time complexity: O(1)
        Space complexity: O(1)
        '''
        return dist[parent_node.node_id] <= dist[child_node.node_id]    # O(1)

    def deleteMin(self, dist):
        '''Time complexity: O(log|V|)
           Space complexity: O(|V|)'''
        node = self.pop(0)  # O(1)
        if len(self) > 1:
            self.insert(0, self[-1])    # O(1)
            self.pop(-1)    # O(1)
            self.heapify(0, dist)   # O(log|V|)
        return node


# queue = HeapPQ()
# for i in range(8):
#     node = CS312GraphNode(i, QPointF(i, i))
#     queue.append(node)
# print('node id:', [node.node_id for node in queue])
#
# test = 2
# if test == 1:
#     print('\nHeapify test 1: dist(parent of bottom node) > dist(bottom node)')
#     dist = [2, 3, 4, 9, 6, 7, 8, 1]
#     print('  dist:', dist)
#     queue.heapify(7, dist)
#     print('  node id:', [node.node_id for node in queue])
# elif test == 2:
#     print('\nHeapify test 2: dist(top node) > dist(children of top node)')
#     dist = [9, 2, 4, 3, 6, 7, 8, 5]
#     print('  dist:', dist)
#     queue.heapify(0, dist)
#     print('  node id:', [node.node_id for node in queue])


class NetworkRoutingSolver:
    def __init__(self):
        self.dist = []
        self.prev = []

    def initializeNetwork(self, network):
        assert(type(network) == CS312Graph)
        self.network = network

    def getShortestPath(self, destIndex):
        # TODO: RETURN THE SHORTEST PATH FOR destIndex
        #       INSTEAD OF THE DUMMY SET OF EDGES BELOW
        #       IT'S JUST AN EXAMPLE OF THE FORMAT YOU'LL 
        #       NEED TO USE
        '''
        Time complexity: O(|V|)
        Space complexity: O(|V|)
        '''

        graph_nodes = self.network.getNodes()

        self.dest = destIndex

        path_edges = []
        total_length = 0
        current_id = self.dest                          # start the tracing at the destination node
        while self.prev[current_id] is not None:        # while the current node has a previous node, O(|V|)
            previous_id = self.prev[current_id]         # get the previous node
            edges = graph_nodes[previous_id].neighbors  # get the previous node's edges
            for edge in edges:                          # iterate through the edges, O(3) = O(1)
                edge_length = edge.length
                if edge.dest.node_id == current_id:     # if this edge connects the previous node to the current node
                    '''add the edge to the list of edges in the path'''
                    path_edges.append((edge.src.loc, edge.dest.loc, '{:.0f}'.format(edge_length)))
                    total_length += edge_length         # increment the path length
                    current_id = previous_id            # move on to the next node in the path

        return {'cost': total_length, 'path': path_edges}

    def computeShortestPaths(self, srcIndex, use_heap=False):
        # TODO: RUN DIJKSTRA'S TO DETERMINE SHORTEST PATHS.
        #       ALSO, STORE THE RESULTS FOR THE SUBSEQUENT
        #       CALL TO getShortestPath(dest_index)
        '''
        Time complexity (array): O(|V|^2)
        Time complexity (heap): O(|V|log|V|)
        Space complexity (both): O(|V|)
        '''

        self.source = srcIndex  # get the index of the starting node

        t1 = time.time()

        graph_nodes = self.network.getNodes()   # get the list of all nodes in the graph

        '''initialize all graph nodes' distances from the starting node to infinity'''
        self.dist = [float('inf') for _ in range(len(graph_nodes))]     # O(|V|)
        self.prev = [None for _ in range(len(graph_nodes))]     # O(|V|)
        self.dist[srcIndex] = 0     # the starting node's distance to itself is zero

        if use_heap:
            queue = HeapPQ()
            for node in graph_nodes:    # O(|V|)
                queue.insertNode(node, self.dist)   # populate the priority queue with graph nodes, O(log|V|)
        else:
            queue = ArrayPQ()
            for node in graph_nodes:    # O(|V|)
                queue.insertNode(node)      # populate the priority queue with graph nodes, O(1)
 
        while len(queue) > 0:   # O(|V|)
            '''get the node in the queue that's the closest to the starting node'''
            current_node = queue.deleteMin(self.dist)   # array: O(|V|); heap: O(log|V|)
            for edge in current_node.neighbors:     # O(3)
                adjacent_node = edge.dest
                edge_len = edge.length

                if self.dist[adjacent_node.node_id] > self.dist[current_node.node_id] + edge_len:
                    self.dist[adjacent_node.node_id] = self.dist[current_node.node_id] + edge_len
                    self.prev[adjacent_node.node_id] = current_node.node_id
                    try:
                        '''heapify if the priority queue's underlying data structure is a heap'''
                        queue.heapify(queue.index(adjacent_node), self.dist)    # O(log|V|)
                    except AttributeError:
                        pass

        t2 = time.time()

        return t2 - t1

