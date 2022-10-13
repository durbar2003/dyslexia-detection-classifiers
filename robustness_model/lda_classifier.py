#importing the libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df=pd.read_excel("./content/speech_data_second.xlsx")

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

model = LinearDiscriminantAnalysis()
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
print("Accuracy_Score : ",round(accuracy_score(Y_test,y_pred)*100,2),"%")