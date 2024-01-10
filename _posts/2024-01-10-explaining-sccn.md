---
title: "understanding 'Efficient Representation Learning for Higher-Order Data with Simplicial Complexes' part 1"
layout: post
---

this post is made with the intent of helping me and my team understand the code that is being used in the above [paper](https://openreview.net/pdf?id=nGqJY4DODN), because in typical ML fashion the code is in a jupyter notebook with minimal explanations. i do not want to discredit the authors, the code is well written, it is just hard to understand and scary looking if you are approaching it for the first time.

this post will mostly focus on helping me understand the code, so it will probably not be perfect, but hopefully it helps somehow. it is handy to have the [code](https://github.com/ruocheny/SCCN-LoG) beside you as you read this.

## building the dataset

### 1. loading the data
```python
''' extract features, labels and graph from the dataset (ref: https://github.com/tkipf/pygcn) '''

idx_features_labels = np.genfromtxt("./data/cora/cora.content",dtype=np.dtype(str))
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
labels = encode_onehot(idx_features_labels[:, -1])
# build graph
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
edges_unordered = np.genfromtxt("./data/cora/cora.cites",dtype=np.int32)
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                    dtype=np.int32).reshape(edges_unordered.shape)
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(labels.shape[0], labels.shape[0]),
                    dtype=np.float32)
# directed --> symmetric adjacency matrix
# in binary case, this is equivalent as adj = (adj + adj.T)/2
# if the network is weighted, only a single edge is kept (the one with largest weight).
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
```
this implementation uses the cora dataset which is a standard dataset used in graph application benchmarking. this particular block of code is creating the adjecency matrix for the cora graph, and i will conveniently use chatgpt to help explain what is going on.

pt 1 - loading features & labels
```python
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
```
> The features are extracted from the loaded data. The features are stored in a sparse matrix using SciPy's csr_matrix with data type np.float32. The features are taken from columns 1 to the second-to-last column of the loaded data.

```python
labels = encode_onehot(idx_features_labels[:, -1])
```
> The labels are extracted from the last column of the loaded data. The function encode_onehot is assumed to convert categorical labels into one-hot encoding.

pt.2 - building the graph
```python
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
```
> The node indices are extracted and converted into a NumPy array. A mapping (idx_map) is created to map the original node indices to consecutive integers.

```python
edges_unordered = np.genfromtxt("./data/cora/cora.cites", dtype=np.int32)
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                dtype=np.int32).reshape(edges_unordered.shape)
```
> The edges of the graph are loaded from the "cora.cites" file. The idx_map is used to map the original node indices to consecutive integers. The reshaping is done to maintain the original shape of the edges array.

```python
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                  shape=(labels.shape[0], labels.shape[0]),
                  dtype=np.float32)
```
> The adjacency matrix (adj) is constructed using the edges information. It is a sparse matrix with ones at the positions corresponding to the edges between nodes.

```python
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
```
> This step ensures that the adjacency matrix is symmetric. For a binary case (unweighted graph), it is equivalent to averaging the matrix with its transpose. If the network is weighted, only the edge with the largest weight is kept.

by adding the original adjacency matrix to its transpose while accounting for double counting, and then subtracting the double-counted edges, the code ensures that the final adjacency matrix is symmetric. This is important for undirected graphs, as the adjacency matrix should reflect the symmetry of the relationships between nodes.

### 2. buiding simplicial complexes

this step seems to be specific to thhe project, as defining how a simplex exists in a graph depends on the data. in this case, if all nodes in a clique (edge, triangle etc.) have same label, it is considered as a simplex. the code for this is as follows:
```python
''' build simplicial complexes from cliques '''
G = nx.from_scipy_sparse_array(adj)
# if all nodes in a clique (edge, triangle etc.) have same label, it is considered as a simplex
# note that we remove esdges whose ends have different labels
SC = []
SC_labels = []
for clq in nx.clique.find_cliques(G):
    if (labels[clq] == labels[clq][0]).all(): # set of nodes have same label
        SC.append(set(clq))
        SC_labels.append(labels[clq][0])
p_max = max([len(SC[i]) for i in range(len(SC))]) - 1
print('number of edges in graph is', G.number_of_edges())
print('number of simplices is', len(SC))
print('maximum dimension of simlices is', p_max)
```
in this code block, the important part is the for loop.
```python
for clq in nx.clique.find_cliques(G):
    if (labels[clq] == labels[clq][0]).all(): # set of nodes have same label
        SC.append(set(clq))
        SC_labels.append(labels[clq][0])
```
> The code iterates over cliques in the graph using nx.clique.find_cliques(G). For each clique (clq), it checks if all nodes in the clique have the same label. If they do, the clique is considered a simplex, and it's added to the SC list as a set of nodes, and its label is added to the SC_labels list.

next, we will be using the following code to build the p-chains as described in the paper. here, a p-chain is a list of p-simplices. the code is easy to understand.
```python
''' build p-chains '''
simplex_chains = [[] for _ in range(p_max+1)]   # simplex_chains[p] is the list of p-chains
simplex_labels = [[] for _ in range(p_max+1)]   # simplex_chains[p] is the list of p-chains' labels
simplex_features = [[] for _ in range(p_max+1)] # simplex_chains[p] is the list of p-chains' features
feat_dense = np.array(features.todense())
for i, s in enumerate(SC):
    simplex_chains[len(s)-1].append(s)
    simplex_labels[len(s)-1].append(SC_labels[i])
    simplex_features[len(s)-1].append(sum(feat_dense[list(s),:],0))
# add 0-simplex to the chains
simplex_chains[0] = [set([ss]) for ss in points]
simplex_labels[0] = labels[list(points),:]
simplex_features[0] = feat_dense[list(points),:]
```

the next part of the code is essential for building the "complete" adjecency matrix and the working of SCCN. in essense, what it does is that it takes a p-chain, and adds all the p-simplices present in that p-chain to another (p-1) chain of lower order p-1 simplices. thus, the final p-1 chain is a concatanation of p-1 simplices as well as the p-simplices. the subsets of these p-simplices here are known as "pseudo simplices".

> Pseudo-simplices, in the context of this code, refer to subsets of higher-dimensional simplices that are introduced to lower-dimensional simplicial complexes. Specifically, these pseudo-simplices are subsets of p-simplices (combinations of p+1 nodes) added to (p−1)-chains.

```python
''' add faces to lower dimension simplex '''
if flag_add_face:
    # chains with pseudo simplex c^p and s(c^{p+1})
    simplex_chains_all = [[] for _ in range(p_max+1)]
    simplex_labels_all = [[] for _ in range(p_max+1)]
    simplex_features_all = [[] for _ in range(p_max+1)]
    for p in range(p_max+1):
        if p < p_max:
            simplex_chains_all[p], simplex_labels_all[p], simplex_features_all[p] \
            = add_pseudo(p+1, simplex_chains[p+1], simplex_labels[p+1], simplex_features[p+1], 
                simplex_chains[p], simplex_labels[p], simplex_features[p])
        else:
            simplex_chains_all[p], simplex_labels_all[p], simplex_features_all[p] \
            = simplex_chains[p].copy(), simplex_labels[p].copy(), simplex_features[p].copy()
```

the add_pseudo function is as follows:
```python
def add_pseudo(p, chain_p, label_p, feature_p, chain, label, feature):
    """ chain_p: p-chain
        chain: (p-1)-chain
        pseudo: (p-1)-pseudo simplex, subset of p-simplex
        chain_w_pseudo: union (p-1)-chain and (p-1)-pseudo simplex"""
    chain_w_pseudo = chain.copy()
    label_w_pseudo = label.copy()
    feature_w_pseudo = feature.copy()
    for s_idx, s in enumerate(chain_p):
        for i in itertools.combinations(s,p):
            if not set(i) in chain_w_pseudo:
                """pseudo simplex (subset of the p-simplex s) doesn't exist before in the (p-1)-chains
                simply add the pseudo simplex"""
                chain_w_pseudo.append(set(i))
                label_w_pseudo.append(label_p[s_idx])
                feature_w_pseudo.append(feature_p[s_idx])
    return chain_w_pseudo, label_w_pseudo, feature_w_pseudo
```

the very nice chatgpt explanation for this is as follows:

Function Parameters:
> p: The dimension of the pseudo-simplices to be added.
> chain_p: The (p)-chain (list of p-simplices) to select pseudo-simplices from.
> label_p: Labels associated with each p-simplex in chain_p.
> feature_p: Features associated with each p-simplex in chain_p.
> chain: The (p-1)-chain (list of (p-1)-simplices) to which pseudo-simplices will be added.
> label: Labels associated with each (p-1)-simplex in chain.
> feature: Features associated with each (p-1)-simplex in chain.

Initialization:
> chain_w_pseudo, label_w_pseudo, and feature_w_pseudo are initialized as copies of the input chain, label, and feature lists, respectively. These lists will be modified to include the pseudo-simplices.

Iteration through (p)-chain:
> The function iterates through the (p)-chain represented by chain_p. s_idx is the index of the current p-simplex s in chain_p.

Generate Pseudo-Simplices:
> It uses itertools.combinations to generate all possible subsets of size p from the vertices of the current p-simplex s.
The code checks if each generated subset is not already present in the modified (p-1)-chain chain_w_pseudo.

Add Pseudo-Simplices:
> If a subset is not in chain_w_pseudo, it is added to chain_w_pseudo.
> The corresponding label and feature information from label_p and feature_p are also appended to label_w_pseudo and feature_w_pseudo.

> In summary, the add_pseudo function takes a (p)-chain, extracts all possible pseudo-simplices of dimension p from each p-simplex, and adds them to a given (p-1)-chain while preserving associated labels and features. The modified (p-1)-chain, labels, and features are then returned. This process is often used in algebraic topology and homological algebra when building chain complexes and studying topological spaces.

### 3. building the incidence matrices, laplacians and the adjacency matrices

this part of the code is consistent with the paper and builds the above matrices according the equations described in the paper. the code is as follows:

```python
def higher_order_ana(p_max, simplex_chains):
    """build incidence matrices"""
    incidences = [[] for _ in range(p_max+2)]
    incidences[0] = np.zeros((1,len(simplex_chains[0])))           # incidence[0] = [0,...,0] (row)
    incidences[p_max+1] = np.zeros((len(simplex_chains[p_max]),1)) # incidence[p_max+1] = [0,...,0] (column)
    for p in range(1,p_max+1):
        # incidences[p]: chain[p] --> chain[p-1]
        incidences[p] = np.zeros((len(simplex_chains[p-1]),len(simplex_chains[p])))
        for i in range(len(simplex_chains[p-1])):
            for j in range(len(simplex_chains[p])):
                if set(simplex_chains[p-1][i]).issubset(set(simplex_chains[p][j])): incidences[p][i][j] = 1

    """build higher order laplacian matrices"""
    laplacians = [[] for _ in range(p_max+1)] # laplacians[p]: p-order laplacian matrix, p=0,...,p_max
    for p in range(p_max+1):
        laplacians[p] = incidences[p].T @ incidences[p] + incidences[p+1] @ incidences[p+1].T

    """extract higher order adjacency matrices from the laplacians"""
    degrees = [np.diag(np.diag(laplacians[i])) for i in range(len(laplacians))]
    adjacencies = laplacians.copy()
    adj_norm_sp = []
    for p in range(len(adjacencies)):
        a_self = 1 if (p==0 or p== p_max) else 2
        np.fill_diagonal(adjacencies[p],a_self) # add self-loops with weight 2: A = A + 2I_N
        adj_norm = normalize(adjacencies[p]) #D^(-1/2)AD^(1/2)
        adj_norm_sp.append(torch.from_numpy(adj_norm).to_sparse().double()) # convert to sparse tensor
        
    return adjacencies, adj_norm_sp, incidences, laplacians
```

### 4. building the "complete" adjecency matrix

the below code defines the weight of connections between different simplices within the same dimension and between different dimensions. the code is as follows:

```python
# connection weight (strength) for simplex self connection; 
# simplex self connection within same dimension; 
# between different dimension (i.e., whether is face and coface)
w_self_con = 1
w_simplex_con = 1
w_face_con = 0.5

for p in range(len(adj_p)):
    adj_p[p] = adj_p[p].to_dense()
    incidence_p[p] = incidence_p[p].to_dense()
    adj_p[p] *= adj_p[p] * w_simplex_con
    adj_p[p].fill_diagonal_(w_self_con)
```
finally, using the definitions and process followed in the paper, the full adjecency matrix is built as follows:

```python
# build connection of all simplex with different dimension
N_p = [feat_p[p].shape[0] for p in range(len(feat_p))]
N = sum(N_p)
adj = torch.zeros((N,N))
for i in range(len(adj_p)):
    adj[sum(N_p[:i]):sum(N_p[:i+1]),sum(N_p[:i]):sum(N_p[:i+1])] = adj_p[i]
    if i < len(adj_p) - 1:
        adj[sum(N_p[:i]):sum(N_p[:i+1]),sum(N_p[:i+1]):sum(N_p[:i+2])] \
        = w_face_con * incidence_p[i+1]
        adj[sum(N_p[:i+1]):sum(N_p[:i+2]),sum(N_p[:i]):sum(N_p[:i+1])] \
        = w_face_con * incidence_p[i+1].T      
adj_sp = adj.to_sparse().double()
```

and that is it for building the features and the adjecency matrix. the next part of the code is the actual SCCN model, which i will be covering in the next post.