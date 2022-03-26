###### Write Your Library Here ###########
from collections import deque
from functools import reduce
from os import startfile
from itertools import permutations , combinations
from queue import PriorityQueue
import math as m
import copy



#########################################


def search(maze, func):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_four_circles": astar_four_circles,
        "astar_many_circles": astar_many_circles
    }.get(func)(maze)


# -------------------- Stage 01: One circle - BFS Algorithm ------------------------ #

def bfs(maze):
    """
    [문제 01] 제시된 stage1의 맵 세가지를 BFS Algorithm을 통해 최단 경로를 return하시오.(20점)
    """
    start_point=maze.startPoint()

    path=[]

    ####################### Write Your Code Here ################################
    n, m = maze.getDimensions()
    q = deque()
    check = [[False]*m for _ in range(n)]
    dist = [[0]*m for _ in range(n)]
    q.append(start_point)
    check[start_point[0]][start_point[1]] = True
    dist[start_point[0]][start_point[1]] = 1

                              
    p = [[[0 for k in range(2)] for j in range(m+1)] for i in range(n+1)]

    while q:
        x, y = q.popleft()
        if maze.isObjective(x,y):
            sx,sy = x, y;
            break
        for nx,ny in maze.neighborPoints(x,y):
            if check[nx][ny] == False:
                q.append((nx,ny))
                dist[nx][ny] = dist[x][y] + 1
                check[nx][ny] = True
                p[nx][ny] = (x,y)
     

    curx = sx
    cury = sy
    while(1):
        path.append((curx,cury))
        if curx == start_point[0] and cury == start_point[1]:
            break;
        curx, cury= p[curx][cury]
    path.reverse()

    return path

    ############################################################################



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
        return self.g+self.h<other.g+other.h 

    def __gt__(self, other):
        return self.g+self.h>other.g+other.h 

    def __ge__(self, other):
        return self.g+self.h>=other.g+other.h 


# -------------------- Stage 01: One circle - A* Algorithm ------------------------ #

def manhatten_dist(p1,p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def astar(maze):

    """
    [문제 02] 제시된 stage1의 맵 세가지를 A* Algorithm을 통해 최단경로를 return하시오.(20점)
    (Heuristic Function은 위에서 정의한 manhatten_dist function을 사용할 것.)
    """

    start_point=maze.startPoint()

    end_point=maze.circlePoints()[0]

    path=[]

    ####################### Write Your Code Here ################################
    
    fringe = []
    closed = []

    start = Node(None,start_point)
    end = Node(None,end_point)

    fringe.append(start)
    
    while fringe:
        cur = fringe[0]

        for node in fringe:
            if node.f < cur.f:
                cur = node
        
        fringe.remove(cur)
        closed.append(cur)

        if cur == end:
            real = cur
            while real is not None:
                x, y = real.location
                path.append(real.location)
                real = real.parent
            path.reverse()
            break

        children = []

        for nx, ny in maze.neighborPoints(cur.location[0],cur.location[1]):
            next = Node(cur,(nx,ny))
            children.append(next)

        for child in children:
            if child in closed:
                continue
            child.g = cur.g + 1
            child.h = manhatten_dist(child.location,end.location)
            child.f = child.g + child.h

            for node in fringe:
                if(child == node and child.g > node.g):
                    continue
            
            fringe.append(child)

    return path

    ############################################################################


# -------------------- Stage 02: Four circles - A* Algorithm  ------------------------ #



def stage2_heuristic(p1,p2):
#    return max(abs(p1[0] - p2[0]) , abs(p1[1] - p2[1])); #diagonal,dominance distance
#    return min(abs(p1[0] - p2[0]) , abs(p1[1] - p2[1]));
#     return 2*(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])); #weighted manhattan 
#    return (abs(p1[0]-p2[0]) / abs(p1[0]+p2[0])) + (abs(p1[1]-p2[1]) / abs(p1[1]+p2[1])); #canberra distance
#    return (abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])) / ((p1[0] + p2[0]) + (p1[1] + p2[1])); #Bray-Curtis distance
#    return  1.2 * ( ( (abs((p1[0] -p2[0])) ** 3) + (abs((p1[1]-p2[1])) ** 3)) ** (1.0/3.0)  )#mincowski distance : generalization of manhattan distance
#    return 1.31*((((p1[0] -p2[0]) ** 2) + ((p1[1]-p2[1]) ** 2)) ** (1.0/2.0)) #euclidian distance  610
#    return 1.31 * m.sqrt( ((p1[0] -p2[0]) ** 2) + ((p1[1]-p2[1]) ** 2) )  # euclidian distance 610
    return ((p1[0] -p2[0]) ** 2) + ((p1[1]-p2[1]) ** 2)
#    return m.sqrt( abs(m.sqrt(p1[0]) - m.sqrt(p2[0])) + abs(m.sqrt(p1[1]) - m.sqrt(p2[1])) ) #chord distance
#    pass
#    dx = abs(p1[0]-p2[0])
#    dy = abs(p1[1] - p2[1])
#    d = 1
#    root = 2 ** 0.5
#    return d * (dx + dy) + (root - 2 * d) * min(dx,dy) 
#    return 1.0*(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])); #manhattan, city block distance 

def astar_four_circles(maze):
    """
    [문제 03] 제시된 stage2의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage2_heuristic function을 직접 정의하여 사용해야 한다.)
    """

    end_points=maze.circlePoints()
    end_points.sort()

    path=[]

    ####################### Write Your Code Here ################################
    n,m = maze.getDimensions()
    shuffle = permutations(end_points)
    start_point = maze.startPoint()

    
    fpathlen = 10000000
    for permu in shuffle:
        
        start_points = []
        start_points.append(start_point[0]); start_points.append(start_point[1]);
        fpath = []

        for s,t in permu:
            fringe = []
            closed = []

            start = Node(None,(start_points[0],start_points[1]))
            end = Node(None,(s,t))

            fringe.append(start)
    
            while fringe:
                cur = fringe[0]

                for node in fringe:
                    if node.f < cur.f:
                        cur = node
        
                fringe.remove(cur)
                closed.append(cur)

                if cur == end:
                    temp = []
                    real = cur
                    while real is not None:
                        x, y = real.location
                        temp.append(real.location)
                        real = real.parent
                    temp.pop()
                    temp.reverse()
                    fpath = fpath + temp
                    break

                children = []

                for nx, ny in maze.neighborPoints(cur.location[0],cur.location[1]):
                    next = Node(cur,(nx,ny))
                    children.append(next)

                for child in children:
                    if child in closed:
                        continue
                    child.g = cur.g + 1
                    child.h = stage2_heuristic(child.location,end.location)
                    child.f = child.g + child.h

                    for node in fringe:
                        if(child == node and child.g > node.g):
                            continue
            
                    fringe.append(child)

            start_points[0] = s; start_points[1] = t;
        
        fpath.insert(0,start_point)
        if len(fpath) < fpathlen:
            fpathlen = len(fpath)
            path = fpath    

    return path

    ############################################################################



# -------------------- Stage 03: Many circles - A* Algorithm -------------------- #
def edge_weights(maze,p1,p2):
    start_point = p1
    end_point = p2
    dx = [0,0,1,-1]
    dy = [1,-1,0,0]
    n, m = maze.getDimensions()
    q = deque()
    check = [[False]*m for _ in range(n)]
    dist = [[0]*m for _ in range(n)]
    q.append(start_point)
    check[start_point[0]][start_point[1]] = True
    dist[start_point[0]][start_point[1]] = 0

    while q:
        x, y = q.popleft()
        if (x,y) is p2:
            break
        for k in range(4):
            nx, ny = x+dx[k], y+dy[k]
            if 0<= nx < n and 0 <= ny < m:
                if check[nx][ny] == False and maze.isWall(nx,ny) == False:
                    q.append((nx,ny))
                    dist[nx][ny] = dist[x][y] + 1
                    check[nx][ny] = True

    
    return dist[end_point[0]][end_point[1]]

def mst(objectives, edges):

    cost_sum=0
    ####################### Write Your Code Here ################################
    approx = 1.07
    parent = {} 
    rank = {}
  
    for i in objectives:
        rank[i] = 0 
        parent[i] = i
    
    def find(a): 
        if parent[a] == a: 
            return a
        p = find(parent[a]) 
        parent[a] = p 
        return parent[a]

    def union(a, b): 
        a = find(a) 
        b = find(b) 
        if a == b: 
            return
        if rank[a] > rank[b]: 
            parent[b] = a 
        else:
            parent[a] = b 
            if rank[a] == rank[b]: 
                rank[b] += 1 

    def kruskal(edges):
        edges.sort() 
        total = 0 
        for edge in edges:
            if not edge: 
                continue
            cost = edge[0]; a = edge[1]; b = edge[2]; 
            if find(a) != find(b): 
                union(a,b) 
                total += cost 
 

        return total

    cost_sum = kruskal(edges) * approx

    return cost_sum
    ############################################################################

def stage3_heuristic(p1,p2):
    dx = abs(p1[0]-p2[0])
    dy = abs(p1[1] - p2[1])
    d = 1
    root = 2 ** 0.5
    return d * (dx + dy) + (root - 2 * d) * min(dx,dy) 
def astar_many_circles(maze):
    """
    [문제 04] 제시된 stage3의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage3_heuristic function을 직접 정의하여 사용해야 하고, minimum spanning tree
    알고리즘을 활용한 heuristic function이어야 한다.)
    """

    end_points= maze.circlePoints()
    end_points.sort()

    path=[]

    ####################### Write Your Code Here ################################
    n, m = maze.getDimensions()
    start_point = maze.startPoint()
    start_points = []
    start_points.append((start_point[0],start_point[1]))

    pathfinder = start_points + end_points

    edge_weight = []
    edge_weight_for_g = {}
    comb = combinations(pathfinder,2)
    for p1,p2 in comb:
        weight = edge_weights(maze,p1,p2)
        edge_weight.append([weight,p1,p2])
#        edge_weight.append([weight,p2,p1])
        edge_weight_for_g[p1,p2] = weight
        edge_weight_for_g[p2,p1] = weight

    fringe = []
    closed = []
    x = start_point[0]
    y = start_point[1]
    root = Node([x,y],[x,y])
    root.parent = root
    root.obj = root.parent
    closed.append(root)
    mypath = []
    pq = PriorityQueue()
#    heap = []

    while (1):
        cur = closed[-1]
        
        h = 0
        for nx,ny in end_points:
             
            if [nx,ny] == cur.location :
                continue

            node = Node(cur,[nx,ny]) 
            node.obj = cur
            sx, sy = cur.location
            node.g = (node.parent).g + edge_weight_for_g[(sx,sy),(nx,ny)]

            
            temp =[]
            nodecopy = node
            while(node.parent != node):
                s,t = node.location
                if (s,t) in temp:
                    node = node.parent
                    continue
                temp.append((s,t))
                node = node.parent
            s,t= node.location
            if (s,t) not in temp:
                temp.append((s,t))
            temp.pop(0)
            node = nodecopy
            part = copy.deepcopy(pathfinder)
            edge_weight_copy = copy.deepcopy(edge_weight)
            for loc in temp:
                part.remove(loc)
                for edge in edge_weight:
                    if edge[1] == loc or edge[2] == loc:
                        if edge in edge_weight_copy:
                            edge_weight_copy.remove(edge)
                
            node.h = mst(part,edge_weight_copy)
            node.f = node.g + node.h

            pq.put(node)
                                                            
        nomad2=pq.get()
        closed.append(nomad2)
        
        the_end = 1
        ending = nomad2
            
        lenser = []
        while(ending.parent!=ending):
            if ending.location not in lenser:
                the_end = the_end+1
            lenser.append(ending.location)
            ending = ending.parent
        
        if the_end == len(pathfinder):
            real = nomad2
            while(real.parent != real):
                x,y=real.location
                mypath.append([x,y])
                real = real.parent
            break

    mypath.reverse()
    start_points = []
    start_points.append(start_point[0]); start_points.append(start_point[1]);
    for s,t in mypath:
        fringe = []
        closed = []

        start = Node(None,(start_points[0],start_points[1]))
        end = Node(None,(s,t))

        fringe.append(start)
    
        while fringe:
            cur = fringe[0]

            for node in fringe:
                if node.f < cur.f:
                    cur = node
        
            fringe.remove(cur)
            closed.append(cur)

            if cur == end:
                temp = []
                real = cur
                while real is not None:
                    x, y = real.location
                    temp.append(real.location)
                    real = real.parent
                temp.pop()
                temp.reverse()
                path = path + temp
                break

            children = []

            for nx, ny in maze.neighborPoints(cur.location[0],cur.location[1]):
                next = Node(cur,(nx,ny))
                children.append(next)

            for child in children:
                if child in closed:
                    continue
                child.g = cur.g + 1
                child.h = stage3_heuristic(child.location,end.location)
                child.f = child.g + child.h

                for node in fringe:
                    if(child == node and child.g > node.g):
                        continue
            
                fringe.append(child)

        start_points[0] = s; start_points[1] = t;
        
    path.insert(0,start_point)

    return path
    ############################################################################
