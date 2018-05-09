
# coding: utf-8

# # FastMap Tutorial

# ## Introduction
# 
# In this tutorial, we will introduce [the FastMap algorithm](http://repository.cmu.edu/cgi/viewcontent.cgi?article=1577&context=compsci) for mapping objects with known pairwise distances into $k$-dimensional spaces. We will introduce the basic working principle of the algorithm and illustrate 3 applications of it. In the first application, we will show the preservation of dissimilarities after FastMap by mapping words with Levenshtein distances. Then, we will show the application of FastMap in dimensionality reduction and cluster visualization through the [WINE](http://archive.ics.uci.edu/ml/datasets/Wine) data set. Finally, we will use a [Wikipedia](http://en.wikipedia.org) corpus to illustrate the application of FastMap in natural language processing.
# 

# ### Table of Contents
# 
# - [Introduction](#Introduction)
# 
# The tutorial consists of two main parts. The first part introduces the FastMap algorithm.
# - [FastMap: the Algorithm](#FastMap:-the-Algorithm)
#    - [Problem Statement](#Problem-Statement)
#    - [General Idea](#General-Idea)
#    - [Pivot Choosing](#Pivot-Choosing)
#    - [The Algorithm](#FastMap:-the-Algorithm)
# 
# In the second part, we will illustrate 3 applications of the FastMap algorithm.
# - [Mapping with Levenshtein Distance](#Mapping-with-Levenshtein-Distance)
# - [Wine Cultivars: Dimensionality Reduction](#Wine-Cultivars:-Dimensionality-Reduction)
# - [Wikipedia Articles Clustering](#Wikipedia-Articles-Clustering)
# 
# Other parts include the following.
# - [Conclusions and Further Readings](#Conclusions-and-Further-Readings)
# - [References](#References)

# ## FastMap: the Algorithm
# 
# [FastMap](http://repository.cmu.edu/cgi/viewcontent.cgi?article=1577&context=compsci) is a fast algorithm proposed in 1995 by Faloutsos and Lin from University of Maryland, College Park to map objects with known pairwise distances to a $k$-dimensional space. In this part, we will introduce the basic working principle of this algorithm.
# 

# ### Problem Statement
# 
# In some cases, data scientists can only have access to <cite>the similarity/distance of two objects</cite> without a well-defined <cite>feature extraction function</cite> [1]. With only the distances between the objects, we can neither perform much useful data analysis and retreival nor visualize the data in a 2d or 3d space. With this in mind, it is important to develop a fast algorithm to map such data with only pairwise distances known to a $k$-dimensional space, where we can perform data analysis and visualization.
# 
# A well-defined problem statement for FastMap is <cite>to find $N$ points in $k$-d space, whose Euclidean distances will match the distances of a given $N\times N$ distance matrix</cite> [1].
# 

# ### General Idea
# 
# The basic idea of FastMap is to carefully select a line and project all the objects onto it for each dimensions. The best projection (preserves the dissimilarities the best) is given when a line $O_aO_b$ through 2 pivot objects $O_a$ and $O_b$ with the farthest distance apart is chosen.
# 
# Suppose we have found such line $O_aO_b$, the way to project an object $O_i$ onto the line is through the [Law of Cosines].
# 
# $$d_{b,i}^2=d_{a,i}^2+d_{a,b}^2-2x_id_{a,b}\qquad\Rightarrow\qquad x_i=\frac{d_{a,i}^2+d_{a,b}^2-d_{b,i}^2}{2d_{a,b}}.$$
# 
# This is illustrated by the following figure [1].
# 
# ![Law of Cosines](cosine_law.png)
# 
# This gives us the mapping of an object to 1 dimension $x_i$. To map to other dimensions, we need to find the projection of the pairwise distances to the hyper-plane perpendicular to the line $O_aO_b$. Let us consider the following figure [1].
# 
# ![Hyper Plane](hyper_plane.png)
# 
# From the figure, it is clear that the new distance between $O_i$ and $O_j$ is given by
# 
# $$d_{i,j}^{'2}=d_{i,j}^2-(x_i-x_j)^2,\qquad\forall i,j\in[N].$$
# 
# Then, by using the new pairwise distances given by $d'$, we can choose a new pair of pivot objects and do a mapping to a second dimension. By performing this process $k$ times recursively, we can finally project the objects into a $k$-dimensional space.
# 

# ### Pivot Choosing
# 
# To choose the objects with the farthest distance apart as pivots would require inspection of each $O(N^2)$ pairwise distances between objects. To do this in each recursion renders the process rather slow.
# 
# A randomized solution to this problem proposed is to first randomly choose a point and find the point $O_a$ farthest apart from it, and then find the point $O_b$ farthest apart from $O_a$. The resulting pivots $O_a$ and $O_b$ are not necessarily the best pivots (those with the farthest distance apart), but works quit well for the FastMap algorithm. Since this process take only $O(N)$ time, it is much better than always choosing the best pair of pivots.
# 

# In[5]:


import numpy as np

def choose_pivot(dist, N):
    
    # 1) Choose arbitrarily an object, and declare it to be the second pivot object Ob
    Ob = np.random.randint(N)
    
    # 2) set Oa = (the object that is farthest apart from Ob) (according to the distance function dist)
    maximum = -1
    for i in range(N):
        d = dist(Ob, i)
        if d > maximum:
            maximum = d
            Oa = i
    
    # 3) set Ob = (the object that is farthest apart from Oa)
    for i in range(N):
        d = dist(Ob, i)
        if d > maximum:
            maximum = d
            Ob = i

    # 4) report the objects Oa and Ob as the desired pair of objects
    if Oa < Ob:
        return Oa, Ob
    else:
        return Ob, Oa


# ### The Algorithm
# 
# The algorithm of FastMap then follows directly from the general idea and the pivot choosing heuristic introduced above. We give an implementation of the algorithm as follows.
# 

# In[6]:


import math

class dist_:
    def __init__(self, D):
        self.col = 0
        self.X = None
        self.D = D
        
    def update(self, X):
        self.X = X
        self.col += 1
        
    def dist(self, i, j):
        dsqr = self.D[i][j] ** 2
        for c in range(self.col):
            dsqr -= (self.X[i][c] - self.X[j][c]) ** 2
        
        if dsqr > 0:
            return math.sqrt(dsqr)
        else:
            return 0

def Mapping(k, D, N, col, X, PA):
    
    if col >= k:
        return X, PA

    # choose pivot objects
    Oa, Ob = choose_pivot(D.dist, N)
    
    # record the ids of the pivot objects
    PA[0][col] = Oa
    PA[1][col] = Ob

    if(D.dist(Oa, Ob) == 0):
        # because all inter-object distances are zeros
        return X, PA

    # project the objects on the line (Oa, Ob)
    for i in range(N):
        X[i][col] = (D.dist(Oa, i) ** 2 + D.dist(Oa, Ob) ** 2 - D.dist(Ob, i) ** 2) / (2 * D.dist(Oa, Ob))

    # consider the projections of the objects on a hyper-plane perpendicular to the line (Oa, Ob)
    D.update(X)

    return Mapping(k, D, N, col + 1, X, PA)

def FastMap(k, D, N):
    
    # At the end of the algorithm, the i-th row will be the image of the i-th object.
    X = np.zeros((N, k))
    # stores the ids of the pivot objects - one pair per recursive call
    PA = np.zeros((2, k))
    D_ = dist_(D)
    # points to the column of the X[] array currently being updated
    col = 0
    
    return Mapping(k, D_, N, col, X, PA)

# To visualize the clustering results, we define a `show_cluster` function which works for both 2d and 3d spaces.

# In[48]:

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_cluster(X, clt, d):
    """
    Show 3 clusters in the 2d or 3d space as a scatter plot.

    Args:
        X:   an embedding of instances to the 2d or 3d space.
        clt: a list of the cluster number from 0 to 2.
        d:   2 or 3. The dimension of the scatter plot.
    """
    cmap = plt.cm.hsv
    colors = cmap(np.linspace(0,1,max(clt) + 1))
    if d == 2:
        for item in zip(X, clt):
            plt.scatter(item[0][0], item[0][1], c=colors[item[1]])
        
    if d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for item in zip(X, clt):
            ax.scatter(item[0][0], item[0][1], item[0][2], c=colors[item[1]])

    plt.show()

def distance(X):
    """
    Return a matrix of dissimilarities indicated by the cosine similarity.
    
    Args:
        X: sparse matrix of TFIDF scores or term frequencies
    
    Returns:
        M: dense numpy array of all pairwise dissimilarities.
    """
    Xd = X.todense()
    
    CS = np.zeros((len(Xd), len(Xd)))
    for doc1 in range(len(Xd)):
        for doc2 in range(len(Xd)):
            if doc1 == doc2:
                CS[doc1][doc2] = np.nan
            else:
                CS[doc1][doc2] = Xd[doc1] @ Xd[doc2].T / np.linalg.norm(Xd[doc1]) / np.linalg.norm(Xd[doc2])
            
    return 2 / math.pi * np.arccos(CS)
