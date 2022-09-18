#importing the libraries

from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pyswarms as ps
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from numpy.random import rand

df=pd.read_excel("./content/speech_data.xlsx")

df.head()

df.shape

df.isnull().sum()

df.iloc[67] # getting certain rows

"""**0.1.1 1. Cleaning Data**"""

clean_df=pd.DataFrame([],columns=df.columns)

for i in range(67):
  clean_df=clean_df.append(df.iloc[i])

for i in range(68,100):
  clean_df=clean_df.append(df.iloc[i])

for i in range(101,df.shape[0]):
  clean_df=clean_df.append(df.iloc[i])

clean_df.shape

clean_df['tarp'].iloc[56]="tarp"

clean_df.head(2)

"""**0.1.2  2. Checking if their is any null values**"""

print(clean_df.isnull().sum().sort_values(ascending=False))

"""**0.1.3 3. Filling Empty Value**"""

clean_df=clean_df.fillna(method='ffill')

print(clean_df.isnull().sum().sort_values(ascending=False)[:5])

"""**0.1.4 4. Seperating datasets**"""

original_words=clean_df.columns
original_words=original_words[1:]
print(original_words)
print(len(original_words))

"""**0.1.5 Removing numbers from strings**"""

import re # regular expression to remove irelevant numbers and symbols
temp=" ".join(original_words)
temp=temp.lower()
original_words=re.findall("[a-zA-Z]+",temp)
print(len(original_words))

# converting to lower case

for i in range(0,len(original_words)):
  clean_df[original_words[i]]=clean_df[original_words[i]].str.lower()
# Another way clean_df['column_name'].apply(lambda x : x.lower())

"""**0.1.6 Calculating Errors and creating models**"""

uttered_words=clean_df.values
uttered_words=uttered_words[:clean_df.shape[0],1:]
m=uttered_words.shape
print(m)
print(uttered_words)

from collections import defaultdict as dt
from collections import Counter

def hammingDist(str1, str2):
    i = 0
    count = 0
 
    while(i < len(str1)):
        if(str1[i] != str2[i]):
            count += 1
        i += 1
    return count


class dyslexia:

  def __init__(self,m):

    self.total_words=m
    # getting error for total student
    self.migrational_error={}
    self.substitutional_error={}
    self.N_error={}


  # Checking if the words are anagram of each other
  def anagram(self,str1,str2):
    if(len(str1)!=len(str2)):
      return False
    str2=str2.lower()
    x=0
    for i in range(len(str1)):
      x=x^ord(str1[i])
      x=x^ord(str2[i])
    if(x==0):
      return True
    else:
      return False

  # Total Error in the given database
  def totalError(self,original_words,uttered_words):
    m_error=0
    s_error=0
    n_error=0
    m=uttered_words.shape[0]
    for i,orig in enumerate(original_words):
      for uw in range(m):
        # No errors if words are pronounced as same
        if (orig==uttered_words[uw:uw+1,i][0].lower()):
          continue
        # migration error if words are anagram
        elif ((len(orig)==len(uttered_words[uw:uw+1,i][0]))):
          if (self.anagram(orig,uttered_words[uw:uw+1,i][0])):
            m_error+=1
          elif (hammingDist(orig,uttered_words[uw:uw+1,i][0]) == 1):
            s_error+=1
          else:
            n_error+=1
        else:
          n_error+=1
    return m_error,s_error,n_error
  
  # passing original words ,ith student uttered words,student number(1,length of original words)
  def error(self,original_words,student,stud_n):
    m_error=0
    s_error=0
    n_error=0

    # for every student append the error
    self.migrational_error['Student %d'%(stud_n)]=dt(list)
    self.substitutional_error['Student %d'%(stud_n)]=dt(list)
    self.N_error['Student %d'%(stud_n)]=dt(list)

    for i,orig in enumerate(original_words):
      # No errors if words are pronounced as same
      if (orig==student[i]):
        continue
      # migational error if words are anagram
      elif ((len(orig)==len(student[i]))):
        if (self.anagram(orig,student[i])):
          self.migrational_error['Student %d'%(stud_n)][orig].append(student[i])
          m_error+=1
        elif (hammingDist(orig,student[i]) == 1):
          self.substitutional_error['Student %d'%(stud_n)][orig].append(student[i])
          s_error+=1
        else:
          n_error+=1
          self.N_error['Student %d'%(stud_n)][orig].append(student[i])
      else:
        n_error+=1
        self.N_error['Student %d'%(stud_n)][orig].append(student[i])
    return m_error,s_error,n_error
  
  def percentage(self,student_no,isprint=False):
    m=len(self.migrational_error['Student %d'%(student_no)])
    s=len(self.substitutional_error['Student %d'%(student_no)])
    n=len(self.N_error['Student %d'%(student_no)])
    if isprint:
      print("Migrational Error % is {}".format((m/(self.total_words)*100)))
      print("Substitutional Error % is {}".format((s/(self.total_words)*100)))
      print("N-Error % is {}".format((n/(self.total_words)*100)))
      if((m==0 and s==0) or (m==0 and n==0) or (s==0 and n==0)):
        print(f"Student {student_no} is Progressing Good or has less dyslexia problem")
    return (m/(self.total_words)*100),(s/(self.total_words)*100),(n/(self.total_words)*100)

# Creating object of dyslexic student
dys=dyslexia(len(original_words))

print("Total Errors in the given Dataset :- ")
m_error,s_error,n_error=dys.totalError(original_words,uttered_words)
print(f'Migrational Error : {m_error},\nSubstitutional Error : {s_error},\nN-Error : {n_error}')

"""**0.1.7 Calculating Error for All student**"""

for stud in range(uttered_words.shape[0]):
  dys.error(original_words,uttered_words[stud],stud+1)

print("Error for student 1")
m_error_1,s_error_1,n_error_1=dys.error(original_words,uttered_words[0],1)
print(f'Migrational Error : {m_error_1},\nSubstitutional Error :␣ {s_error_1},\nN-Error : {n_error_1}')

print("What are the errors that student 1 made : -")
print(f"Migrational Error : \n{dys.migrational_error['Student 1']}")

print("What are the errors that student 1 made : -")
print(f"Substitutional Error : \n{dys.substitutional_error['Student 1']}")

print("What are the errors that student 1 made : -")
print(f'N Error : \n{dys.N_error["Student 1"]}')

print("Percentage Error for student 1 : ")
dys.percentage(1)

dys.percentage(2,True)

fig,ax=plt.subplots(5,4,figsize=(28,25))
fig.suptitle('Comparison 1st 20 students : ')
labels=["Migrational Error","Substitutional Error","N-Error"]
for i in range(ax.shape[0]):
  for j in range(ax.shape[1]):
    ax[i][j].set_title("Student %d"%(i*ax.shape[1]+j+1))
    ax[i][j].bar(labels[0],((len(dys.migrational_error["Student %d"%(i*ax.shape[1]+j+1)])/60)*100),width=0.4,label=labels[0])
    ax[i][j].bar(labels[1],((len(dys.substitutional_error["Student %d"%(i*ax.shape[1]+j+1)])/60)*100),width=0.4,label=labels[1])
    ax[i][j].bar(labels[2],((len(dys.N_error["Student %d"%(i*ax.shape[1]+j+1)])/60)*100),width=0.4,label=labels[2])
plt.savefig("Comparison_20_students")

dys.percentage(20) ## Error percentage of 20th student

train,test=train_test_split(clean_df,test_size=0.3)

print(train.shape)
print(test.shape)

train.head()

migrational_error=[]
substitutional_error=[]
N_Error=[]
for i in range(clean_df.shape[0]):
  a,b,c=dys.percentage(i+1)
  migrational_error.append(a)
  substitutional_error.append(b)
  N_Error.append(c)

new_df=pd.DataFrame({"migrational_error": migrational_error,"substitutional_error":substitutional_error,'N_Error': N_Error})

new_df.head()

new_df['Avg']=new_df.mean(axis=1)

new_df.head()

new_df['Dyslexic']=(new_df['Avg']>=16)*1

new_df.head()

"""**0.1.8 0 - Not Dyslexic**

**0.1.9 1 - Dyslexic**
"""

new_df=new_df.drop(['Avg'],axis=1)

train,test=train_test_split(new_df,test_size=0.3,random_state=2)

print(train.shape,test.shape)

X=train.values
Y=test.values

X_train=X[:,:-1]
Y_train=X[:,-1]
X_test=Y[:,:-1]
Y_test=Y[:,-1]


global_features = []
labels          = []
seed      = 9
# error rate
def error_rate(xtrain, ytrain, x, opts):
    # parameters
    k     = opts['k']
    fold  = opts['fold']
    num_trees=opts['num_trees']
    seeds=['seeds']
    xt    = fold['xt']
    yt    = fold['yt']
    xv    = fold['xv']
    yv    = fold['yv']
    
    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    xtrain  = xt[:, x == 1]
    ytrain  = yt.reshape(num_train)  # Solve bug
    xvalid  = xv[:, x == 1]
    yvalid  = yv.reshape(num_valid)  # Solve bug   
    # Training
    #mdl     = KNeighborsClassifier(n_neighbors = k)
    mdl  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
    mdl.fit(xtrain, ytrain)
    # Prediction
    ypred   = mdl.predict(xvalid)
    acc     = np.sum(yvalid == ypred) / num_valid
    error   = 1 - acc
    
    return error


# Error rate & Feature size
def Fun(xtrain, ytrain, x, opts):
    # Parameters
    alpha    = 0.99
    beta     = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
    else:
        # Get error rate
        error = error_rate(xtrain, ytrain, x, opts)
        # Objective function
        cost  = alpha * error + beta * (num_feat / max_feat)
        
    return cost



def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X


def init_velocity(lb, ub, N, dim):
    V    = np.zeros([N, dim], dtype='float')
    Vmax = np.zeros([1, dim], dtype='float')
    Vmin = np.zeros([1, dim], dtype='float')
    # Maximum & minimum velocity
    for d in range(dim):
        Vmax[0,d] = (ub[0,d] - lb[0,d]) / 2
        Vmin[0,d] = -Vmax[0,d]
        
    for i in range(N):
        for d in range(dim):
            V[i,d] = Vmin[0,d] + (Vmax[0,d] - Vmin[0,d]) * rand()
        
    return V, Vmax, Vmin


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x
    

def pso(xtrain, ytrain, opts):
    # Default Parameters
    ub    = 1
    lb    = 0
    thres = 0.5
    w     = 0.9    # inertia weight
    c1    = 2      # acceleration factor
    c2    = 2      # acceleration factor
    
    N        = opts['N']
    max_iter = opts['T']
    if 'w' in opts:
        w    = opts['w']
    if 'c1' in opts:
        c1   = opts['c1']
    if 'c2' in opts:
        c2   = opts['c2'] 
    
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
    
    # Initialize position & velocity
    X             = init_position(lb, ub, N, dim)
  
    V, Vmax, Vmin = init_velocity(lb, ub, N, dim) 
    
    # Pre
    fit   = np.zeros([N, 1], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='float')
    fitG  = float('inf')
    Xpb   = np.zeros([N, dim], dtype='float')
    fitP  = float('inf') * np.ones([N, 1], dtype='float')
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    #Stopping Criteria
    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if fit[i,0] < fitP[i,0]:
                Xpb[i,:]  = X[i,:]
                fitP[i,0] = fit[i,0]
            if fitP[i,0] < fitG:
                Xgb[0,:]  = Xpb[i,:]
                fitG      = fitP[i,0]
        
        Gbin2       = binary_conversion(Xgb, thres, 1, dim) 
        Gbin2       = Gbin2.reshape(dim)
        pos2        = np.asarray(range(0, dim))
        sel_index2  = pos2[Gbin2 == 1]
        
       
        print("Feature Subset after {} iterations {}: ".format(t+1,sel_index2))
        
        # Store result
        curve[0,t] = fitG.copy()
        
        print("Iteration:", t + 1)
        
        print("Best (PSO):", curve[0,t])
        t += 1
        
        
        for i in range(N):
            for d in range(dim):
                # Update velocity
                r1     = rand()
                r2     = rand()
                V[i,d] = w * V[i,d] + c1 * r1 * (Xpb[i,d] - X[i,d]) + c2 * r2 * (Xgb[0,d] - X[i,d]) 
                # Boundary
                V[i,d] = boundary(V[i,d], Vmin[0,d], Vmax[0,d])
                # Update position
                X[i,d] = X[i,d] + V[i,d]
                # Boundary
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
           
                
    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    pso_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return pso_data    

gasf=global_features

label=global_labels

# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(gasf, label, test_size=0.3, random_state=0, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# # parameter
# k    = 5     # k-value in KNN
# N    = 10    # number of particles
# T    = 100   # maximum number of iterations
# w    = 0.9   # inertia weight
# c1   = 2     # cognitive factor
# c2   = 2     # social factor 

# parameter
k    = 8     # k-value in KNN
num_trees=100  #random forest
seeds=9       #rf
N    = 20    # number of particles
T    = 10   # maximum number of iterations
w    = 0.5   # inertia weight
c1   = 2     # cognitive factor
c2   = 2     # social factor 
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'w':w, 'c1':c1, 'c2':c2, 'num_trees':num_trees, 'seeds':seeds}

# perform feature selection
fmdl = pso(gasf, label, opts)
sf   = fmdl['sf']
print("fmdl",fmdl);

# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train) 
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  

#Modelling using KNN
#mdl       = KNeighborsClassifier(n_neighbors = k) 

#Modelling using Random Forest
mdl  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
mdl.fit(x_train, y_train)

# Accuracy
y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("Accuracy:", 100 * Acc)

# number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)
n_sf=[]
print(sf)

# plot convergence
curve   = fmdl['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('PSO')
ax.grid()
plt.show()

