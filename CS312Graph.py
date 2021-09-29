#!/usr/bin/python3


class CS312GraphEdge:
    def __init__(self, src_node, dest_node, edge_length):
        '''
        type(src_node): CS312GraphNode
        type(dest_node): CS312GraphNode
        type(edge_length): int
        '''
        self.src = src_node
        self.dest = dest_node
        self.length = edge_length

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '(src={} dest={} length={})'.format(self.src, self.dest, self.length)


class CS312GraphNode:
    def __init__(self, node_id, node_loc):
        '''
        type(node_id): int
        type(node_loc): QPointF
        type(neighbors): list (CS312GraphEdge)
        '''
        self.node_id = node_id  # stores the index location of the node in the graph
        self.loc = node_loc     # stores the coordinate of the node
        self.neighbors = []     # stores the node's out edges (should be 3 of them)

    def addEdge(self, neighborNode, weight):
        self.neighbors.append(CS312GraphEdge(self, neighborNode, weight))

    def __str__(self):
        neighbors = [edge.dest.node_id for edge in self.neighbors]
        return 'Node(id:{},neighbors:{})'.format(self.node_id, neighbors)


class CS312Graph:
    def __init__(self, nodeList, edgeList):
        '''
        nodeList (type: list (QPointF)): a list of node loccations
        edgeList (e.g. {0: [(1, 2), (2, 3), (3, 4)]}): a dictionary of node id (keys) and (node id, weight) (values)
        type(nodes): list (CS312GraphNode)
        '''
        self.nodes = []
        for i in range(len(nodeList)):
            self.nodes.append(CS312GraphNode(i, nodeList[i]))   # create (node id, node loc) and append nodes

        for i in range(len(nodeList)):
            neighbors = edgeList[i]     # [(x1, y1), (x2, y2), (x3, y3)]
            for n in neighbors:
                self.nodes[i].addEdge(self.nodes[n[0]], n[1])   # create (neighborNode, weight) and append edges
        
    def __str__(self):
        s = []
        for n in self.nodes:
            s.append(n.neighbors)
        return str(s)

    def getNodes(self):
        return self.nodes

