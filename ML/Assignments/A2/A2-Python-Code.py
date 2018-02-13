
# coding: utf-8

# <h1> Assignment 2 </h1>

# Name: Hritik Soni
# 
# ID: 2014A2PS480P
# 
# Email: f2014480@pilani.bits-pilani.ac.in
# 
# Dataset Information:
# 
# 45000 Rows 62 Cols
# 
# Type: csv
# 
# Target Type: Nominal (0 or 1)
# 
# Since the target is nominal, we cannot use linear regression.
# 
# We will attempt to use the following classifiers:
# 
# 1. KNN - We will use k = sqrt(# of instances) as it is expected to give decent results.
# 
# 2. D-Trees
# 3. Random Forest
# 4. Naive Bayes
# 5. MLP
# 
# First step is to import all required libraries.

# In[174]:


print("Importing Libraries...")

import math
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
#SVC was taking too much time
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


# Let us set some globals related to our dataset.
# 

# In[175]:


dims = 61 #Known number of attributes
instances = 45000 #number of datapoints

classifiers = ["knn", "dtrees", "rforest", "nb", "mlp"]#This classifiers in this list will be used for model selection
#Classfier can be one of these - knn, dtrees, rforest, nb, mlp

k = int(math.sqrt(instances))#Sets k for knn classifier

numiter = 3 #Number of iterations per classifier for determining score


# In[176]:


print("Loading Dataset")
dataset = np.loadtxt("dataA2.csv", delimiter=",").astype(int)
X, y = dataset[:, 0:61], dataset[:, 61]


# Let us try to visualize the data in two dimensions and see if it is linearly separable or some kind of observable pattern is present.

# In[177]:


dim2 = PCA(n_components=2)
dim2.fit(X)
X3 = dim2.fit_transform(X)


# In[178]:


dfplot = pd.DataFrame(X3, columns=['x', 'y'])
dfplot.plot.scatter('x', 'y', c=y)


# From above plot, we can see that the data can be discretized well especially in the second dimension and thus decision trees or forests can be expected to perform well.

# In[179]:


print("Analyzing Data Using PCA ...")
for i in range(1,dims):
   pca = PCA(n_components=i)
   pca.fit(X)
   print (i, "\t:", sum(pca.explained_variance_ratio_))


# From above decomposition, it is easy to see that we capture just over 99% data by considering 34 dimensions.
# Therefore, for speed reasons and avoiding overfitting we can consider 34 dimensions instead of 61.

# In[180]:


print("Reducing data dimensions to 34")
pca = PCA(n_components=34)
pca.fit(X)
X2 = pca.fit_transform(X)


# Now, we have to split the data into testing and training data. We will use 10% of the data for testing.

# In[181]:


results = {}

for c in classifiers:
    print("Training using classifier:", c)
    results[c] = []
    for itr in range(numiter):
        X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.1)
        if c == "knn":
            classifier = KNeighborsClassifier(n_neighbors=k)
        elif c == "dtrees":
            classifier = DecisionTreeClassifier()
        elif c == "nb":
            classifier = GaussianNB()
        elif c == "rforest":
            classifier = RandomForestClassifier(n_estimators=45, max_depth = 20)
        elif c == "mlp":
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            classifier = MLPClassifier(hidden_layer_sizes=(34, 34, 34),max_iter=100)
        classifier.fit(X_train, y_train)
        results[c].append(classifier.score(X_test, y_test))
        print("Done")
#         print("Training Score: ", classifier.score(X_train, y_train))
#         print("Testing Score: ", results[c][-1])
    


# The code below will print results.

# In[182]:


print("Here are the results:")
print()

maxscore = -math.inf
chosenc = None

for c in classifiers:
    print("Using Classifier:", c)
    print()
    for itr in range(numiter):
        print("Execution %s Score:"%(itr), results[c][itr])
    thisavg = sum(results[c])/numiter
    if thisavg > maxscore:
        maxscore = thisavg
        chosenc = c
    print("Execution Average Score:", thisavg)
    print()
    
print("The maximum score was obtained by classifer: %s"%(chosenc))
print("The score was", maxscore)
print("There we should pick classifier: %s"%(chosenc), "as our base model.")


# Considering Random Forests as our base model, we can improve the accuracies by fiddling with tree depth, number of trees, etc.
# The following code rigorously checks validation accuracy for various depths and tree counts.

# In[183]:


PERFORM_1 = False #Set this to True to perform all the tests

cycles = 3

max_train = -math.inf
maxtr_n = None
maxtr_d = None

max_test = -math.inf
maxte_n = None
maxte_d = None

for n_trees in [1, 43, 44, 45, 46, 47, 100]:
    if not PERFORM_1:
        break
    for depth in [10, 18, 19, 20, 21, 22, 30]:
        print("Trees %s Depth %s"%(n_trees, depth))
        print()
        X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.1)
        this_tescoreavg = 0
        this_trscoreavg = 0
        for c_i in range(cycles):
            classifier = RandomForestClassifier(n_estimators=n_trees, max_depth = depth)
            classifier.fit(X_train, y_train)
            thistescore = classifier.score(X_test, y_test)
            thistrscore = classifier.score(X_train, y_train)
            this_tescoreavg += thistescore
            this_trscoreavg += thistrscore
            print("Cycle %s Train %s Test %s"%(c_i, thistrscore, thistescore))
        this_tescoreavg /= cycles
        this_trscoreavg /= cycles
        if this_tescoreavg > max_test:
            maxte_n = n_trees
            maxte_d = depth
            max_test = this_tescoreavg
        if this_trscoreavg > max_train:
            maxtr_n = n_trees
            maxtr_d = depth
            max_train = this_trscoreavg
        print("Average Train Score: %s"%(this_trscoreavg))
        print("Average Test Score: %s"%(this_tescoreavg))
        print()
            
if PERFORM_1:
    print()        
    print("Best Train Score: %s Trees %s Depth %s"%(max_train, maxtr_n, maxtr_d))
    print("Best Test Score: %s Trees %s Depth %s"%(max_test, maxte_n, maxte_d))


# Over several fold execution of the above code the best validation score was obtained at Tree Depth = 20 and Trees = 45.

# <h2> Result Summary</h2>
# 

# Classifier: Random Forest Classifier (inside sklearn.ensemble)
# Additional Tuning: max_depth = 20 num_estimators = 46
# 
# Accuracy Details:
# 
# Cycle 0 Train 0.8164197530864198 Test 0.6433333333333333
# Cycle 1 Train 0.8147654320987654 Test 0.642
# Cycle 2 Train 0.816 Test 0.646
# Average Train Score: 0.8157283950617283
# Average Test Score: 0.6437777777777778
