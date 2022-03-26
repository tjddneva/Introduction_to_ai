from collections import deque
from itertools import permutations, combinations
import copy
from queue import PriorityQueue
from anytree import Node, RenderTree
a = [(1,2),[3,1],[2,0],[9,9]]
b = [[1,3]]

#temp = combinations(a,2)

#for i,j in temp:
#    print(i,j)

class Node:
    def __init__(self,parent,location):
        self.parent=parent
        self.location=location #현재 노드

        self.obj=[]

        # F = G+H
        self.f=0
        self.g=0
        self.h=0

    def __eq__(self, other):
        return self.location==other.location and str(self.obj)==str(other.obj)

    def __le__(self, other):
        return self.g+self.h<=other.g+other.h

    def __lt__(self, other):
        return self.g+self.h<=other.g+other.h

    def __gt__(self, other):
        return self.g+self.h>other.g+other.h

    def __ge__(self, other):
        return self.g+self.h>=other.g+other.h

fringe = []
pq = PriorityQueue()
node1 = Node(None,[1,2])
node2 = Node(None,[2,3])
node3 = Node(None,[3,4])

node1.g = 1; node1.h = 1
node2.g = 1; node2.h = 1
node3.g = 1; node3.h = 1
pq.put(node1)
pq.put(node2)
pq.put(node3)
while pq.empty() ==False:
    print(pq.get().location)



