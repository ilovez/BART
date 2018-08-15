import numpy as np
from numpy import exp, log, ones, zeros, sqrt, array, arange, pi
import random
from treelib import Node, Tree
import pandas as pd
from itertools import chain, combinations
from copy import deepcopy
from scipy.stats import norm, gamma
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from numba import jit
from sklearn.linear_model import LinearRegression as OLS
#%%
def f1(x):
    x1,x2 = x
    if np.isin(x2,['a','b']):
        return 8 if x1 <= 5 else 2
    else:
        if x1 <= 3: return 1
        elif x1 <= 7: return 5
        else: return 8        

def genData(n=800,seed=None):
    if seed is not None:
        np.random.seed(seed)
    x1val = arange(1,11)
    x2val = ['a','b','c','d']    
    x1 = np.random.choice(x1val,size=n)
    x2 = np.random.choice(x2val,size=n)
    y = array(list(map(f1,tuple(zip(x1,x2))))) + 2*np.random.normal(size=n)
    return pd.DataFrame({'y':y,'x1':x1,'x2':x2})

def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    n = len(s)
    S = chain.from_iterable(combinations(s, r) for r in range(1,n))
    return [set(i) for i in S]

def get_xvar(df):
    n = df.iloc[:,1:].apply(lambda s:len(set(s)))
    return arange(1,df.shape[1])[n>1]    
    
def prune(tree):
    # only pairs of terminal nodes with no children    
    w2 = tree.w2
    pid = random.choice(w2)
    pNode = tree[pid]
    nChild = array([tree[c].data.shape[0] for c in pNode.fpointer])
    nXS = len(pNode.xvar)*len(pNode.S)    
    d = tree.level(pid)+1
    df = pNode.data
    R0 = tree.R[df.index].sum()**2
    R1 = array([tree.R[tree[c].data.index].sum()**2 for c in pNode.fpointer])
    vp = var+len(df)*var_mu
    vc = var+nChild*var_mu
    p1 = vc.prod()/(var*vp)
    p2 = R0/vp - (R1/vc).sum()
    pGrow = 1 if pid == 0 else Prob[0]
    pPrune = Prob[1]-Prob[0]
    cid = set(pNode.fpointer)
    leaf = set(tree.leaf)
    b = len(leaf-cid)+1
    rLike = sqrt(p1)*exp(0.5*var_mu/var*p2)
    rTransit = pGrow/pPrune*len(w2)/(b*nXS)
    rStruct = (d**beta-alpha)*nXS/(alpha*(1-alpha/(1+d)**beta)**2)
    r = rLike*rTransit*rStruct
#    print(f'{mi} prune {t}: {pNode.tag}; r={r.round(4)}')
    if random.uniform(0,1) < r:        
        for i in cid: 
            if i in leaf: tree.leaf.remove(i)
            tree.remove_node(i)
        pNode.tag = str(pid)
        pNode.var = None
        pNode.S = None
        tree.leaf.append(pid)
        w2.remove(pid)
        try:
            if tree.siblings(pid)[0].var==None: w2.append(pNode.bpointer)
        except IndexError:
            w2 = []
#        tree.show()
    return tree

def grow(tree):    
    #random.seed(98)
    b = len(tree.leaf)
    if b<1: raise ValueError('cannot grow anymore')
    pid = random.choice(tree.leaf) 
    pNode = tree[pid]
    x = random.choice(pNode.xvar)   
    dtype = DataTypes[x]
    df = pNode.data
    xVal = set(df.iloc[:,x])
    if (dtype == 'O') | (dtype == 'u'):
        S = powerset(xVal)
        s = random.choice(S)
        idx = df.iloc[:,x].isin(s)
    elif (dtype == 'i') | (dtype == 'f'):
        S = sorted(xVal)[:-1] #leave the last element out
        s = random.choice(S)
        idx = (df.iloc[:,x] <= s)
    nXS = len(pNode.xvar)*len(S)
    #if nXS <= 0: raise ValueError('nXS <= 0')
    childDF = [df.loc[idx],df.loc[~idx]] 
    nChild = array([df.shape[0] for df in childDF])
    if nChild.min() < minObs: 
        print(f'{mi} grow {t}: {(pid,x,s)}; less than {minObs}')
        return tree    
    R0 = tree.R[df.index].sum()**2
    R1 = array([tree.R[df.index].sum()**2 for df in childDF])
    d = tree.level(pid)+1
    gpid = pNode.bpointer
    w2 = len(tree.w2)
    if gpid not in tree.w2: w2+=1    
    vp = var+len(df)*var_mu
    vc = var+nChild*var_mu
    p1 = var*vp/vc.prod()
    p2 = (R1/vc).sum()-R0/vp
    pGrow = Prob[0]; pPrune = ProbDefault[1]-ProbDefault[0]
    rLike = sqrt(p1)*exp(0.5/var_ratio*p2)
    rTransit = pPrune/pGrow*b*nXS/w2    
    rStruct = alpha*(1-alpha/(1+d)**beta)**2/(d**beta-alpha)/nXS
    r = rLike*rTransit*rStruct
    print(f'{mi} grow {t}: {(pid,x,s)}; r={r.round(4)}')
    if random.uniform(0,1) < r:      
        cid = array([pid,pid])*2+array([1,2])
        nodes = [None,None]
        tree.leaf.remove(pid)
        for i in range(2):
            df = childDF[i]
            nodes[i] = tree.create_node(str(cid[i]),cid[i],parent=pid,data=df)
            xvar = get_xvar(df)
            nodes[i].xvar = xvar
            if len(xvar)>0: tree.leaf.append(cid[i])                
            nodes[i].var = None         
        pNode.tag = str(pid)+'-'+str(x)+'-'+str(s)
        pNode.var = x
        pNode.split = s
        pNode.S = S
        if gpid in tree.w2: tree.w2.remove(gpid)
        tree.w2.append(pid)
        tree.show()
    return tree

def genTree(node,sList=[]):    
    nodeType = type(node)
    tree = Tree()
    if nodeType == pd.core.frame.DataFrame:        
        df = node
        root = tree.create_node(tag='0',identifier=0,data=df)
        root.xvar = get_xvar(df)
    elif nodeType == Node:
        root = deepcopy(node)
        root.fpointer = []
        root.bpointer = []
        df = root.data
        tree.add_node(root)
    w2 = []        
    for split in sList:
        pid,x,s = split
        pNode = tree[pid]
        df = pNode.data
        dtype = DataTypes[x]
        xVal = set(df.iloc[:,x])
        if (dtype == 'O') | (dtype == 'u'):
            if not (s < xVal): raise IndexError('invalid category split')
            S = powerset(xVal)
            idx = df.iloc[:,x].isin(s)
        elif (dtype == 'i') | (dtype == 'f'):
            S = sorted(xVal)[:-1] #leave the last element out
            idx = (df.iloc[:,x] <= s)
        childDF = [df.loc[idx],df.loc[~idx]] 
        nodes = [None,None]
        cid = array([pid,pid])*2+array([1,2])
        for i in range(2):
            if len(childDF[i])<minObs: raise IndexError('no data in leaf')
            nodes[i] = tree.create_node(str(cid[i]),cid[i],parent=pid,data=childDF[i])
            nodes[i].var = None
            nodes[i].xvar = get_xvar(childDF[i])             
        pNode.tag = str(pid)+'-'+str(x)+'-'+str(s)
        pNode.S = S
        pNode.var = x
        pNode.split = s
        gpid = pNode.bpointer
        if gpid in w2: w2.remove(gpid)
        w2.append(pid)
    tree.w2 = w2
    return tree         

# make sure obj data type's ancestor are not in selection
def nidValid(tree):
    internalNodes = [n for n in tree.all_nodes_itr() if n.var != None]
    nObj = (n for n in internalNodes if DataTypes[n.var]=='O') 
    noChange = set()
    for node in nObj:
        current = node
        while current.identifier != tree.root:
            ancestor = tree[current.bpointer]
            if ancestor.var == node.var:
                noChange.add(ancestor.identifier)
                break
            current = ancestor
    nInternal = [n.identifier for n in 
                 filter(lambda n:n.identifier not in noChange, internalNodes)]
    return nInternal

def swap(tree):
    internalNodes = [n for n in tree.all_nodes_itr() if n.var != None]
    if len(internalNodes) == 1: return tree
    internalNodes.remove(tree[0])
    cNode = random.choice(internalNodes)
    tagc = (cNode.identifier,cNode.var,cNode.split)
    pid = cNode.bpointer
    tree1 = Tree(tree,deep=True)
    sub = tree1.remove_subtree(pid)
    tags = recurTag(sub,pid)
    tagp = tags[0]
    tags[tags.index(tagc)] = (tagc[0],tagp[1],tagp[2])
    tags[0] = (tagp[0],tagc[1],tagc[2])
    string = f'{mi} swap {t}: {tags[0]}; '
    try:
        sub1 = genTree(tree[pid],tags)
    except IndexError:
        print(string + 'unswappable')
        return tree
    #rTransit = 1
    rLike = get_like_ratio(tree.R,sub.leaves(),sub1.leaves())
    rStruct = get_struct(sub.all_nodes_itr(),sub1.all_nodes_itr())
    r = rLike*rStruct
    print(string + f'{r.round(4)}')
    if random.uniform(0,1) < r: 
        if pid > 0:
            gpid = tree[pid].bpointer
            tree1.paste(gpid,sub1)
            tree1[gpid].fpointer = sorted(tree1[gpid].fpointer)
        else:
            tree1 = sub1
        tree1.w2 = tree.w2        
        tree1.R = tree.R
        tree1.leaf = [n.identifier for n in tree1.leaves() if len(n.xvar)>0]
        tree1.show()
        return tree1
    return tree
      
def change(tree):    
    nidInternal = nidValid(tree)
    choices = [getChoice(tree,n) for n in nidInternal]
    n_choices = map(lambda L: sum([len(i) for i in L]),choices)
    choiceDic = {a:b for (a,b,c) in zip(nidInternal,choices,n_choices) if c > 1}
    choices1 = list(choiceDic.keys())
    nid = random.choice(choices1)    
    p = tree[nid].data.shape[1]
    x0 = tree[nid].var; s0 = tree[nid].split
    choices = choiceDic[nid] # choose nid to split
    if s0 in choices[x0-1]: choices[x0-1].remove(s0) # remove original split option
    choices2 = [i for i in range(p-1) if len(choices[i])>0] # choose var to split
    x = random.choice(choices2)
    choices3 = choices[x] # choose value to split
    x += 1
    s = random.choice(choices3)
    tree1 = Tree(tree,deep=True)
    pid = tree1[nid].bpointer
    sub = tree1.remove_subtree(nid)
    tags = recurTag(sub,nid)
    tags[0] = (nid,x,s)
    try:
        sub1 = genTree(sub[nid],tags)
    except IndexError:
        print(f'{mi} change {t}: {tags[0]}; unchangable')
        return tree
    if pid is not None:
        tree1.paste(pid,sub1)
        tree1[pid].fpointer = sorted(tree1[pid].fpointer)
    else:
        tree1 = sub1
    nidInternal1 = set(nidValid(tree1))
    choices1 = set(choices1)
    choices11 = nidInternal1.intersection(choices1)
    extra = nidInternal1-choices1
    n_choices = map(lambda L: sum([len(i) for i in L]),
                    [getChoice(tree1,n) for n in extra])
    choices11 = list(choices11)+[a for (a,b) in zip(extra,n_choices) if b > 1]
    choices31 = getChoice(tree1,nid,x0)[x0-1]
    n31 = len(choices31)
    if (sub1[nid].var==sub[nid].var) and (s0 in choices31):
        n31 -=1
    rTransit = len(choices1)*len(choices3)/(len(choices11)*n31)
    rLike = get_like_ratio(tree.R,sub.leaves(),sub1.leaves())
    rStruct = get_struct(sub.all_nodes_itr(),sub1.all_nodes_itr())
    r = rLike*rTransit*rStruct
    print(f'{mi} change {t}: {tags[0]}; r={r.round(4)}')
    if random.uniform(0,1) < r:    
        tree1.w2 = tree.w2        
        tree1.R = tree.R
        tree1.leaf = [n.identifier for n in tree1.leaves() if len(n.xvar)>0]
        tree1.show()
        return tree1
    return tree

def getChoice(tree,nid,x=None):
    df = tree[nid].data
    p = df.shape[1]
    if x == None:
        xvar = arange(1,p)
        choices = [sorted(set(df.iloc[:,i])) for i in xvar]
    elif float(x).is_integer():
        xvar = [x]
        choices = [None]*p
        choices[x-1] = sorted(set(df.iloc[:,x]))    
    childID = tree[nid].fpointer
    ileft = [[] for i in range(p)]
    iright = [[] for i in range(p)] # must be deep copy
    for r in recurTag(tree,childID[0]):
        ileft[r[1]].append(r[2])
    for r in recurTag(tree,childID[1]):
        iright[r[1]].append(r[2])
    
    for i in xvar:
        type_i = DataTypes[i]
        choice_i = choices[i-1]
        llegit = set(choice_i)
        rlegit = llegit
        if len(ileft[i]) > 0:            
            if type_i == 'f' or type_i == 'i':
                minValue = max(ileft[i])      
                choice_i = array(choice_i)
                llegit = set(choice_i[choice_i > minValue])
            elif type_i == 'O' or type_i == 'u':
                choices[i-1] = []
        if len(iright[i]) > 0:            
            if type_i == 'f' or type_i == 'i':
                maxValue = min(iright[i])
                choice_i = array(choice_i)
                rlegit = set(choice_i[choice_i < maxValue])
            elif type_i == 'O' or type_i == 'u':
                choices[i-1] = []
        if type_i == 'f' or type_i == 'i':
            # must leave one value out
            choices[i-1] = sorted(llegit.intersection(rlegit))[:-1] 
        elif (type_i == 'O' or type_i == 'u') and choices[i-1] != []:
            choices[i-1] = powerset(choices[i-1])
    return choices 

def get_struct(sub_internal,sub1_internal):
    nXS = [len(n.xvar)*len(n.S) for n in sub_internal if n.var is not None]
    nXS1 = [len(n.xvar)*len(n.S) for n in sub1_internal if n.var is not None]
    return array(nXS).prod()/array(nXS1).prod()
    
def get_like_ratio(R,leaf,leaf1):
    vn0 = array([len(n.data) for n in leaf]) + var_ratio
    vn1 = array([len(n.data) for n in leaf1]) + var_ratio
    R0 = array([R[n.data.index].sum()**2 for n in leaf])
    R1 = array([R[n.data.index].sum()**2 for n in leaf1])
    p1 = vn0.prod()/vn1.prod()
    p2 = 0.5*lamda*(sum(R1/vn1)-sum(R0/vn0))
    return sqrt(p1)*exp(p2)

def drawM(l):
    idx = l.data.index
    n = l.data.shape[0]    
    var = 1/(n*lamda + tau)
    mu = (n*lamda*tree.R[idx].mean() + taumu)*var
    M = np.random.normal(mu,sqrt(var))
    MM.loc[idx,mi] = M
    l.M = M
    
def recurTag(tree,i):
    node = tree[i]
    childID = node.fpointer
    if childID != []:
        tag = (node.identifier,node.var,node.split)
        return [tag] + recurTag(tree,childID[0]) + recurTag(tree,childID[1])
    return []
        
def drawTree(tree):
    global Prob
    if tree.contains(1):        
        #print(Prob)
        u = random.uniform(0,1)
        Prob = ProbDefault
        if  u < Prob[0]:
            tree = grow(tree)
        elif u < Prob[1]:
            tree = prune(tree)    
        elif u < Prob[2]:
            tree = change(tree)
        else:
            tree = swap(tree)
    else:   
        #print(Prob)
        Prob = [1,1,1]
        tree = grow(tree)    
    return tree
#%%
df = pd.read_csv('SkillCraft1_Dataset.csv').dropna()
df0 = df.iloc[:2000]; df1 = df.iloc[2000:]
y = df0.iloc[:,0]
dfd = pd.get_dummies(df0)
e = y-OLS().fit(dfd.iloc[:,1:],dfd.iloc[:,0]).predict(dfd.iloc[:,1:])
#df0 = df1
#df0 = genData()
#df0['x3'] = 22

#%%
v = 3; q = 1-0.9
k = 2; m = 50 # mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm is here
alpha = 0.95; beta = 2
minObs = 3

n0,p = df0.shape
var = e.var()
lamda = 1/var
ymu = y.mean()
sst = (y-ymu)@(y-ymu)
ig1 = v/2; 
ig2 = minimize(lambda x:abs(gamma.cdf(lamda,ig1,scale=x)-q),x0=lamda*ig1)['x'][0]
ig2 = minimize(lambda x:abs(gamma.cdf(lamda,ig1,scale=x)-q),x0=ig2)['x'][0]
ig2 = 1/ig2
a = ig1+n0*0.5-1
from scipy import stats
gg = gamma.cdf(lamda,ig1,scale=1/ig2)
g = stats.invgamma.cdf(var,ig1,scale=ig2)
mumu = (y.min()+y.max())*0.5/m
sigma_mu = (y.max()-m*mumu)/(k*sqrt(m))
var_mu = sigma_mu**2
tau = 1/var_mu
taumu = tau*mumu

DataTypes = df0.dtypes.map(lambda x:x.kind)
#%%
tree = Tree()
root = tree.create_node('0',0,data=df0)
root.xvar = get_xvar(df0)
root.var = None
tree.w2 = []
tree.leaf = [0]
ProbDefault = array([2.5,2.5,4]).cumsum()/9
## tree = trueTree
T = 1250; burn = 250
trees = [deepcopy(tree) for i in range(m)]
MM = pd.DataFrame(index=df0.index,columns=range(m),data=mumu)
Yhat = zeros((n0,T))
Depth_mu = zeros(T)
tdic = [None for i in range(m*T)]
#tdic = [tdic.copy() for i in range(T)]

def tree2dic(tree):
    return {i.identifier:i.M if i.var is None else (i.var, i.split) 
            for i in tree.all_nodes_itr()}

@jit        
def route(dic,row):
    i = 0
    M = dic[i]
    while type(M) == tuple:
        x,s = M
        dtype = DataTypes[x]
        xval = row[x]
        if (dtype == 'i') | (dtype == 'f'):
            i = 2*i+1 if xval <= s else 2*i+2
        elif (dtype == 'O') | (dtype == 'u'):
            i = 2*i+1 if xval in s else 2*i+2
        M = dic[i]
    return M
#%%
def rowpredict(row):
    y = sum(map(lambda d: route(d,row),tdic))
    return y
#for i in range(m):
#    for l in trees[i].leaves():
#        idx = l.data.index
#        MM.loc[idx,i] = mumu
#%%  
Like = zeros(T)
Rmse = zeros(T)
# = zeros(T) 
for t in range(T):
    var_ratio = var/var_mu 
    for mi in range(m):
        trees[mi].R = y-(MM.sum(axis=1)-MM.iloc[:,mi])
        tree = drawTree(trees[mi])
        g = any(map(drawM,tree.leaves()))
        #for l in tree.leaves(): drawM(l)
        trees[mi] = tree
        tdic[t*m+mi] = tree2dic(tree)
    yhat = MM.sum(axis=1).values
    Yhat[:,t] = yhat
    e = (y-yhat).values
    sse = e@e
    Rmse[t] = sqrt(sse/n0)
    Like[t] = log(norm.pdf(e,scale=sqrt(var))).sum()        
    b = ig2+0.5*sse
    lamda = np.random.gamma(a,1/b)
    var = 1/lamda
    Depth_mu[t] = array([tr.depth() for tr in trees]).mean()

yhat = Yhat[:,burn:].mean(axis=1)
e = y-yhat
L1 = abs(e).sum()
L2 = e@e
rmse = sqrt(L2/n0)
R2 = 1-L2/sst
dep = Depth_mu.mean()
#%%    
plt.plot(Like)  
plt.title('Likelihood')
plt.figure()  
plt.plot(Rmse)
plt.title('RMSE')
plt.figure()  
plt.plot(Depth_mu)
plt.title('Depth_Mu')
#%%
#a = rowpredict(df1.iloc[0,:])
T -= burn
tdic = tdic[burn*m:]
yhat1 = df1.apply(rowpredict,axis=1)/T
y1 = df1.iloc[:,0]
e1 = y1-yhat1
L21 = e1@e1
rmse1 = sqrt(L21/len(y1))
R21 = 1-L21/((y1-y1.mean())@(y1-y1.mean()))

