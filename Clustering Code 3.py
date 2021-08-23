#!/usr/bin/env python
# coding: utf-8

# In[138]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[139]:


#Dataset auslesen
data_org = pd.read_csv("datahealth.csv")
data_org


# In[140]:


#unnötige Spalten löschen

del data_org['id']
del data_org["smoking_status"]
del data_org["work_type"]
del data_org["bmi"]
del data_org["gender"]
del data_org["avg_glucose_level"]
#del data_org["stroke"]
data_org


# In[141]:


age1 = data_org['age']

print(age1)


# In[142]:


#MAximum ALter finden

max_age= max(age1)
print(max_age)


# In[143]:


# ForSchleife implementieren
Age = []
for i in age1:
    norm = i/max_age
    #ptint(norm)
    Age.append(norm)
print(Age)


# In[144]:


#neue Spale mit normierten Alter hinzufügen
data_org.insert(loc=0, column='Age', value=Age)

#Alte Spalte löschen
del data_org["age"]
data_org


# In[145]:


#Trainingsdaten und Testdaten trennen 50% zu 50%

msk = np.random.rand(len(data_org)) < 0.5
data = data_org[msk]
data_test = data_org[~msk]

print(data)


# In[146]:


# Zweier Kombination aus den 5 besten Atributen finden
from itertools import combinations as com
import sklearn.cluster as cluster

#For Schleife implemetierern

attr =0
for tuppel in com(data,2):
    attr+=1
    print(attr)
    print(tuppel)


# In[147]:


#'Kombination darstellen'
kombi1 =data[['Age', 'hypertension']]
kombi2= data[['Age', 'heart_disease']]
kombi3=data[['Age', 'ever_married']]
kombi4=data[['Age', 'Residence_type']]
kombi5= data[['hypertension', 'heart_disease']]
kombi6 = data[['hypertension', 'ever_married']]
kombi7=data[['hypertension', 'Residence_type']]
kombi8= data[['heart_disease', 'ever_married']]
kombi9 = data[['heart_disease', 'Residence_type']]
kombi10=data[['ever_married', 'Residence_type']]


# In[148]:


#Inertai definieren
K1 = range(1,10)
inertia1 =[]
for k1 in K1:
    kmeans= cluster.KMeans(n_clusters = k1, init ="k-means++")
    kmeans = kmeans.fit(kombi1)
    inertia_iter1= kmeans.inertia_
    inertia1.append(inertia_iter1)
    inertia1 = sorted(inertia1,reverse = True)
print(inertia1)

#Inertia normieren'

max1 = max(inertia1)
print(max1)
inertia1_max1=[]
#For Schleife implemetierern

for i in inertia1:
    norm = i/max1
    #print(norm)
    inertia1_max1.append(norm)
print(inertia1_max1)

#Kmeans Plotten
plt.figure(figsize =(15,11))
plt.plot(K1, inertia1_max1, 'x-')
plt.xlabel('K')
plt.ylabel('normierte Inertia Werte')
plt.title('Der Elbow Kurv')
plt.legend(["Age', 'hypertension"])
#plt.show()


# In[149]:


#Inertai definieren
K2 = range(1,10)
inertia2 =[]
for k2 in K2:
    kmeans= cluster.KMeans(n_clusters = k2, init ="k-means++")
    kmeans = kmeans.fit(kombi2)
    inertia_iter2= kmeans.inertia_
    inertia2.append(inertia_iter2)
    inertia2 = sorted(inertia2,reverse = True)
print(inertia2)

#Inertia normieren'

max2 = max(inertia2)
print(max2)
inertia2_max2=[]
#For Schleife implemetierern
for i in inertia2:
    norm = i/max2
    #print(norm)
    inertia2_max2.append(norm)
print(inertia2_max2)

#Kmeans Plotten

plt.figure(figsize =(15,11))
plt.plot(K2, inertia2_max2, 'x-')
plt.xlabel('K')
plt.ylabel('normierte Inertia Werte')
plt.title('Der Elbow Kurv')
plt.legend(["Age', 'heart_disease"])
#plt.show()


# In[150]:


#Inertai definieren
K3 = range(1,10)
inertia3 =[]
for k3 in K3:
    kmeans= cluster.KMeans(n_clusters = k3, init ="k-means++")
    kmeans = kmeans.fit(kombi3)
    inertia_iter3= kmeans.inertia_
    inertia3.append(inertia_iter3)
    inertia3 = sorted(inertia3, reverse = True)
print(inertia3)

#Inertia normieren'

max3 = max(inertia3)
print(max3)
inertia3_max3=[]
#For Schleife implemetierern
for i in inertia3:
    norm = i/max3
    #print(norm)
    inertia3_max3.append(norm)
print(inertia3_max3)

#Elbow Plotten
plt.figure(figsize =(15,11))
plt.plot(K3, inertia3_max3, 'x-')
plt.xlabel('K')
plt.ylabel('normierte Inertia Werte')
plt.title('Der Elbow Kurv')
plt.legend(["Age', 'ever_married"])
plt.show()


# In[151]:


#Inertai definieren
K4 = range(1,10)
inertia4 =[]
for k4 in K4:
    kmeans= cluster.KMeans(n_clusters = k4, init ="k-means++")
    kmeans = kmeans.fit(kombi4)
    inertia_iter4= kmeans.inertia_
    inertia4.append(inertia_iter4)
    inertia4 = sorted(inertia4, reverse = True)
print(inertia4)

#Inertia normieren'

max4 = max(inertia4)
print(max4)
inertia4_max4=[]
#For Schleife implemetierern
for i in inertia4:
    norm = i/max4
    #print(norm)
    inertia4_max4.append(norm)
print(inertia4_max4)

#Kmeans Plotten

plt.figure(figsize =(15,11))
plt.plot(K4, inertia4_max4, 'x-')
plt.xlabel('K')
plt.ylabel('normierte Inertia Werte')
plt.title('Der Elbow Kurv')
plt.legend(["Age', 'Residence_type"])
#plt.show()


# In[152]:


#Unnötigen Daten löschen

del data["heart_disease"]
del data["hypertension"]
del data["Residence_type"]


data


# In[ ]:





# In[153]:


#Cluster zentrum ermitteln

kmeans = sklearn.cluster.KMeans(n_clusters = 2, init='k-means++', random_state =0).fit(data)
kmeans.cluster_centers_


# In[154]:



## Perform clustering using Kmeans

n_clusters = 2
cluster = sklearn.cluster.KMeans(n_clusters, random_state=111)
columns = ["Age","ever_married"]
est = kmeans.fit(data[columns])
clusters = est.labels_
data['cluster'] = clusters

# Print some data about the clusters:

# For each cluster, count the members.
for c in range(n_clusters):
    cluster_members=data[data['cluster'] == c][:]
    print('Cluster{}(n={}):'.format(c, len(cluster_members)))
    print('-'* 17)
print(data.groupby(['cluster']).mean())


# In[155]:


#Einzelnen daten zu jeden Custern zuordnen

df1 = data[data.cluster == 0]
df2 = data[data.cluster == 1]

print(df1)
print(df2)
df1.info()


# In[156]:


plt.scatter(df1.Age, df1['ever_married'], color = "blue")
plt.scatter(df2.Age, df2['ever_married'], color = 'black')

plt.xlabel('Age')
plt.ylabel('ever_married')


# In[157]:


#Predictive  mit Trainingsdaten ermitteln 

train_predicted = kmeans.fit_predict(data[['stroke']])
print(train_predicted[0:500])


# In[158]:


#Predictiv mit Trainingsdaten mit AKtual vergleichen 
data['Y_Predicted']= train_predicted.tolist()
data


# In[159]:


#Traindata in einen neuen DataFrame hinzufügen

df_train = pd.DataFrame(train_predicted, columns=['Train_Predicted'])
df_train


# In[160]:


#Predictiv mit Test daten ermitteln
test_predicted = kmeans.fit_predict(data_test[['stroke']])
#data['Test_Predicted']= test_predicted.tolist()
df= pd.DataFrame(test_predicted, columns=['Test_Predicted'])
df
#test_predicted


# In[161]:


#Confusionsmatrix ermitteln

data = (train_predicted,test_predicted)
datatu


# In[162]:


#Zu string umwandeln und matrix zeigen lassen

a = df_train['Train_Predicted']
b = df['Test_Predicted']
#print(type(a))
#print(a)
confusion_matrix =pd.crosstab(a, b, rownames=['Train_predicted'], colnames=['Test_predicted'])
confusion_matrix


# In[163]:


#Confusionsmatrix plotten

sn.heatmap(confusion_matrix, annot=True)

plt.show()


# In[164]:


#Precisons ermitteln

precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
precision


# In[165]:


np.mean(precision)


# In[166]:


data4= pd.DataFrame(train_predicted, columns=['Train predicted'])
data4.insert(loc=1, column='Test predicted', value=train_predicted)
data4


# In[167]:


from sklearn.metrics import accuracy_score
accuracy_score(data4['Train predicted'],data4['Test predicted'])


# In[168]:


#a = df_train['Train_Predicted']
#b = df['Test_Predicted']
#tp, fn, fp, tn = confusion_matrix(df_train['Train_Predicted'],df['Test_Predicted'],labels=(1,0)).ravel()
#accuracy = (tp + tn) / (tp + tn + fp + fn)


# In[ ]:




