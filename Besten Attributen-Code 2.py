#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


data= pd.read_csv("datahealth.csv")
data


# In[35]:


rest_Atributen = data.columns

rest_Atributen = rest_Atributen.drop("id")

rest_Atributen = rest_Atributen.drop("stroke")

rest_Atributen = rest_Atributen.drop("bmi")

rest_Atributen = rest_Atributen.drop("work_type")

rest_Atributen = rest_Atributen.drop("smoking_status")

print(rest_Atributen)


# In[52]:


from itertools import combinations as com
import sklearn.cluster as cluster


# In[53]:


attr = 0
for tupel in com(rest_Atributen,5):
    attr+=1
    print(attr)
    print(tupel)


# In[54]:


#Unötige Spalten bzw. Attribute löschen
del data['id']
del data["smoking_status"]
del data["work_type"]
del data["bmi"]
del data["stroke"]

data


# In[55]:


# Attributen normieren
scaler = MinMaxScaler()

#Gender
scaler.fit(data[['gender']])
data['gender'] = scaler.transform(data[['gender']])

#Age
scaler.fit(data[['age']])
data['age'] = scaler.transform(data[['age']])

#hypertension

scaler.fit(data[['hypertension']])
data['hypertension'] = scaler.transform(data[['hypertension']])

#heart_disease

scaler.fit(data[['heart_disease']])
data['heart_disease'] = scaler.transform(data[['heart_disease']])

# Ever married
scaler.fit(data[['ever_married']])
data['ever_married'] = scaler.transform(data[['ever_married']])


#Residence_type
scaler.fit(data[['Residence_type']])
data['Residence_type'] = scaler.transform(data[['Residence_type']])

#avg_glucose_level
scaler.fit(data[['avg_glucose_level']])
data['avg_glucose_level'] = scaler.transform(data[['avg_glucose_level']])

data


# In[56]:


#Kombination Darstellen

komb1=data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married']]

komb2=data[['gender', 'age', 'hypertension', 'heart_disease', 'Residence_type']]

komb3=data[['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level']]

komb4=data[['gender', 'age', 'hypertension', 'ever_married', 'Residence_type']]

komb5=data[['gender', 'age', 'hypertension', 'ever_married', 'avg_glucose_level']]

komb6=data[['gender', 'age', 'hypertension', 'Residence_type', 'avg_glucose_level']]

komb7=data[['gender', 'age', 'heart_disease', 'ever_married', 'Residence_type']]

komb8=data[['gender', 'age', 'heart_disease', 'ever_married', 'avg_glucose_level']]

komb9=data[['gender', 'age', 'heart_disease', 'Residence_type', 'avg_glucose_level']]

komb10=data[['gender', 'age', 'ever_married', 'Residence_type', 'avg_glucose_level']]

komb11=data[['gender', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type']]

komb12=data[['gender', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level']]

komb13=data[['gender', 'hypertension', 'heart_disease', 'Residence_type', 'avg_glucose_level']]

komb14=data[['gender', 'hypertension', 'ever_married', 'Residence_type', 'avg_glucose_level']]

komb15=data[['gender', 'heart_disease', 'ever_married', 'Residence_type', 'avg_glucose_level']]

komb16=data[['age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type']]

komb17=data[['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level']]

komb18=data[['age', 'hypertension', 'heart_disease', 'Residence_type', 'avg_glucose_level']]

komb19=data[['age', 'hypertension', 'ever_married', 'Residence_type', 'avg_glucose_level']]

komb20=data[['age', 'heart_disease', 'ever_married', 'Residence_type', 'avg_glucose_level']]

komb21=data[['hypertension', 'heart_disease', 'ever_married', 'Residence_type', 'avg_glucose_level']]


# In[57]:


K1 = range(1,10)
inertia1 = []
for k1 in K1:
    kmeans = cluster.KMeans(n_clusters = k1, init = "k-means++")
    kmeans = kmeans.fit(komb1)
    inertia_iter1 = kmeans.inertia_
    inertia1.append(inertia_iter1)
print(inertia1)


# In[58]:


K2 = range(1,10)
inertia2 = []
for k2 in K2:
    kmeans = cluster.KMeans(n_clusters = k2, init = "k-means++")
    kmeans = kmeans.fit(komb2)
    inertia_iter2 = kmeans.inertia_
    inertia2.append(inertia_iter2)


# In[59]:


K3 = range(1,10)
inertia3 = []
for k3 in K3:
    kmeans = cluster.KMeans(n_clusters = k3, init = "k-means++")
    kmeans = kmeans.fit(komb3)
    inertia_iter3 = kmeans.inertia_
    inertia3.append(inertia_iter3)


# In[60]:


K4 = range(1,10)
inertia4 = []
for k4 in K4:
    kmeans = cluster.KMeans(n_clusters = k4, init = "k-means++")
    kmeans = kmeans.fit(komb4)
    inertia_iter4 = kmeans.inertia_
    inertia4.append(inertia_iter4)


# In[61]:


K5 = range(1,10)
inertia5 = []
for k5 in K5:
    kmeans = cluster.KMeans(n_clusters = k5, init = "k-means++")
    kmeans = kmeans.fit(komb5)
    inertia_iter5 = kmeans.inertia_
    inertia5.append(inertia_iter5)


# In[62]:


K6 = range(1,10)
inertia6 = []
for k6 in K6:
    kmeans = cluster.KMeans(n_clusters = k6, init = "k-means++")
    kmeans = kmeans.fit(komb6)
    inertia_iter6 = kmeans.inertia_
    inertia6.append(inertia_iter6)


# In[63]:


K7 = range(1,10)
inertia7 = []
for k7 in K7:
    kmeans = cluster.KMeans(n_clusters = k7, init = "k-means++")
    kmeans = kmeans.fit(komb7)
    inertia_iter7 = kmeans.inertia_
    inertia7.append(inertia_iter7)


# In[64]:


K8 = range(1,10)
inertia8 = []
for k8 in K8:
    kmeans = cluster.KMeans(n_clusters = k8, init = "k-means++")
    kmeans = kmeans.fit(komb8)
    inertia_iter8= kmeans.inertia_
    inertia8.append(inertia_iter8)


# In[65]:


K9 = range(1,10)
inertia9 = []
for k9 in K9:
    kmeans = cluster.KMeans(n_clusters = k9, init = "k-means++")
    kmeans = kmeans.fit(komb9)
    inertia_iter9 = kmeans.inertia_
    inertia9.append(inertia_iter9)


# In[66]:


K10 = range(1,10)
inertia10 = []
for k10 in K10:
    kmeans = cluster.KMeans(n_clusters = k10, init = "k-means++")
    kmeans = kmeans.fit(komb10)
    inertia_iter10 = kmeans.inertia_
    inertia10.append(inertia_iter10)


# In[67]:


K11 = range(1,10)
inertia11 = []
for k11 in K11:
    kmeans = cluster.KMeans(n_clusters = k11, init = "k-means++")
    kmeans = kmeans.fit(komb11)
    inertia_iter11 = kmeans.inertia_
    inertia11.append(inertia_iter11)


# In[68]:


K12 = range(1,10)
inertia12 = []
for k12 in K12:
    kmeans = cluster.KMeans(n_clusters = k12, init = "k-means++")
    kmeans = kmeans.fit(komb12)
    inertia_iter12 = kmeans.inertia_
    inertia12.append(inertia_iter12)


# In[69]:


K13 = range(1,10)
inertia13 = []
for k13 in K13:
    kmeans = cluster.KMeans(n_clusters = k13, init = "k-means++")
    kmeans = kmeans.fit(komb13)
    inertia_iter13 = kmeans.inertia_
    inertia13.append(inertia_iter13)


# In[70]:


K14 = range(1,10)
inertia14 = []
for k14 in K14:
    kmeans = cluster.KMeans(n_clusters = k14, init = "k-means++")
    kmeans = kmeans.fit(komb14)
    inertia_iter14 = kmeans.inertia_
    inertia14.append(inertia_iter14)


# In[71]:


K15 = range(1,10)
inertia15 = []
for k15 in K15:
    kmeans = cluster.KMeans(n_clusters = k15, init = "k-means++")
    kmeans = kmeans.fit(komb15)
    inertia_iter15 = kmeans.inertia_
    inertia15.append(inertia_iter15)


# In[72]:


K16 = range(1,10)
inertia16 = []
for k16 in K16:
    kmeans = cluster.KMeans(n_clusters = k16, init = "k-means++")
    kmeans = kmeans.fit(komb16)
    inertia_iter16 = kmeans.inertia_
    inertia16.append(inertia_iter16) 


# In[73]:


K17 = range(1,10)
inertia17 = []
for k17 in K17:
    kmeans = cluster.KMeans(n_clusters = k17, init = "k-means++")
    kmeans = kmeans.fit(komb17)
    inertia_iter17 = kmeans.inertia_
    inertia17.append(inertia_iter17)


# In[74]:


K18 = range(1,10)
inertia18 = []
for k18 in K18:
    kmeans = cluster.KMeans(n_clusters = k18, init = "k-means++")
    kmeans = kmeans.fit(komb18)
    inertia_iter18 = kmeans.inertia_
    inertia18.append(inertia_iter18)


# In[75]:


K19 = range(1,10)
inertia19 = []
for k19 in K19:
    kmeans = cluster.KMeans(n_clusters = k19, init = "k-means++")
    kmeans = kmeans.fit(komb19)
    inertia_iter19 = kmeans.inertia_
    inertia19.append(inertia_iter19)


# In[76]:


K20 = range(1,10)
inertia20 = []
for k20 in K20:
    kmeans = cluster.KMeans(n_clusters = k20, init = "k-means++")
    kmeans = kmeans.fit(komb20)
    inertia_iter20 = kmeans.inertia_
    inertia20.append(inertia_iter20)


# In[77]:


K21 = range(1,10)
inertia21 = []
for k21 in K21:
    kmeans = cluster.KMeans(n_clusters = k21, init = "k-means++")
    kmeans = kmeans.fit(komb21)
    inertia_iter21 = kmeans.inertia_
    inertia21.append(inertia_iter21)


# In[79]:


# Normierten Inertia 1
inertia1 = sorted(inertia1,reverse = True)
print(inertia1)
max1 = max(inertia1)
print(max1)
inertia1_max1=[]
for i in inertia1:
    norm = i/max1
    #print(norm)
    inertia1_max1.append(norm)
print(inertia1_max1)

# Normierten Inertia 2
inertia2 = sorted(inertia2,reverse = True)
print(inertia2)
max2 = max(inertia2)
print(max2)
inertia2_max2=[]
for i in inertia2:
    norm = i/max2
    #print(norm)
    inertia2_max2.append(norm)
print(inertia2_max2)

#Normierte Inertia 3
inertia3 = sorted(inertia3,reverse = True)
print(inertia3)
max3 = max(inertia3)
print(max3)
inertia3_max3=[]
for i in inertia3:
    norm = i/max3
    #print(norm)
    inertia3_max3.append(norm)
print(inertia3_max3)


# In[80]:


plt.figure(figsize =(15,11))
plt.plot(K1, inertia1_max1, 'x-')
plt.plot(K2, inertia2_max2, 'x-')
plt.plot(K3, inertia3_max3, 'x-')
plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('Der Elbow Kurv')
plt.legend(["gender', 'age', 'hypertension', 'heart_disease', 'ever_married","gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level", "gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level"])
plt.show()


# In[81]:


#Normierten Inertia 4
inertia4 = sorted(inertia4,reverse = True)
print(inertia4)
max4 = max(inertia4)
print(max4)
inertia4_max4=[]
for i in inertia4:
    norm = i/max4
    #print(norm)
    inertia4_max4.append(norm)
print(inertia4_max4)

#Normierten Inertia5
inertia5 = sorted(inertia5,reverse = True)
print(inertia5)
max5 = max(inertia5)
print(max5)
inertia5_max5=[]
for i in inertia5:
    norm = i/max5
    #print(norm)
    inertia5_max5.append(norm)
print(inertia5_max5)

#Normierten Inertia6
inertia6 = sorted(inertia6,reverse = True)
print(inertia6)
max6 = max(inertia6)
print(max6)
inertia6_max6=[]
for i in inertia6:
    norm = i/max6
    #print(norm)
    inertia6_max6.append(norm)
print(inertia6_max6)


# In[82]:


plt.figure(figsize =(15,11))
plt.plot(K4, inertia4_max4, 'x-')
plt.plot(K5, inertia5_max5, 'x-')
plt.plot(K6, inertia6_max6, 'x-')
plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('Der Elbow Kurv')
plt.legend(["gender', 'age', 'hypertension', 'ever_married', 'Residence_type","gender', 'age', 'hypertension', 'ever_married', 'avg_glucose_level", "gender', 'age', 'hypertension', 'Residence_type', 'avg_glucose_level"])
plt.show()


# In[83]:


#normiertia Inertia7
inertia7 = sorted(inertia7,reverse = True)
print(inertia7)
max7 = max(inertia7)
print(max7)
inertia7_max7=[]
for i in inertia7:
    norm = i/max7
    #print(norm)
    inertia7_max7.append(norm)
print(inertia7_max7)

#Normierten Inertia8
inertia8 = sorted(inertia8,reverse = True)
print(inertia8)
max8 = max(inertia8)
print(max8)
inertia8_max8=[]
for i in inertia8:
    norm = i/max8
    #print(norm)
    inertia8_max8.append(norm)
print(inertia8_max8)

#Normierten Inertia 9

inertia9 = sorted(inertia9,reverse = True)
print(inertia9)
max9 = max(inertia9)
print(max9)
inertia9_max9=[]
for i in inertia9:
    norm = i/max9
    #print(norm)
    inertia9_max9.append(norm)
print(inertia9_max9)


# In[84]:


plt.figure(figsize =(15,11))
plt.plot(K7, inertia7_max7, 'x-')
plt.plot(K8, inertia8_max8, 'x-')
plt.plot(K9, inertia9_max9, 'x-')
plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('Der Elbow Kurv')
plt.legend(["gender', 'age', 'heart_disease', 'ever_married', 'Residence_type","gender', 'age', 'heart_disease', 'Residence_type', 'avg_glucose_level", "gender', 'age', 'heart_disease', 'ever_married', 'avg_glucose_level"])
plt.show()


# In[85]:


#Normierten Inertia10

inertia10 = sorted(inertia10,reverse = True)
print(inertia10)
max10 = max(inertia10)
print(max10)
inertia10_max10=[]
for i in inertia10:
    norm = i/max10
    #print(norm)
    inertia10_max10.append(norm)
print(inertia10_max10)

#Normierte Inertia 11

inertia11 = sorted(inertia11,reverse = True)
print(inertia11)
max11 = max(inertia11)
print(max11)
inertia11_max11=[]
for i in inertia11:
    norm = i/max11
    #print(norm)
    inertia11_max11.append(norm)
print(inertia11_max11)

#Normierten Inertia 12

inertia12 = sorted(inertia12,reverse = True)
print(inertia1)
max12 = max(inertia12)
print(max12)
inertia12_max12=[]
for i in inertia12:
    norm = i/max12
    #print(norm)
    inertia12_max12.append(norm)
print(inertia12_max12)


# In[86]:


plt.figure(figsize =(15,11))
plt.plot(K10, inertia10_max10, 'x-')
plt.plot(K11, inertia11_max11, 'x-')
plt.plot(K12, inertia12_max12, 'x-')
plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('Der Elbow Kurv')
plt.legend(["gender', 'age', 'ever_married', 'Residence_type', 'avg_glucose_level","gender', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level","gender', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level"])
plt.show()


# In[87]:


#Normierte  Inertia13

inertia13 = sorted(inertia13,reverse = True)
print(inertia13)
max13 = max(inertia13)
print(max13)
inertia13_max13=[]
for i in inertia13:
    norm = i/max13
    #print(norm)
    inertia13_max13.append(norm)
print(inertia13_max13)

#Normierte Inertia14

inertia14 = sorted(inertia14,reverse = True)
print(inertia14)
max14 = max(inertia14)
print(max14)
inertia14_max14=[]
for i in inertia14:
    norm = i/max14
    #print(norm)
    inertia14_max14.append(norm)
print(inertia14_max14)

#Normierte Inertia15

inertia15 = sorted(inertia11,reverse = True)
print(inertia15)
max15 = max(inertia15)
print(max15)
inertia15_max15=[]
for i in inertia15:
    norm = i/max15
    #print(norm)
    inertia15_max15.append(norm)
print(inertia15_max15)


# In[88]:


plt.figure(figsize =(15,11))
plt.plot(K13, inertia13_max13, 'x-')
plt.plot(K14, inertia14_max14, 'x-')
plt.plot(K15, inertia15_max15, 'x-')
plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('Der Elbow Kurv')
plt.legend(["gender', 'hypertension', 'heart_disease', 'Residence_type', 'avg_glucose_level","'gender', 'hypertension', 'ever_married', 'Residence_type', 'avg_glucose_leve","gender', 'heart_disease', 'ever_married', 'Residence_type', 'avg_glucose_level"])
plt.show()


# In[89]:


#Normierten Inetia16
inertia16 = sorted(inertia16,reverse = True)
print(inertia16)
max16 = max(inertia16)
print(max16)
inertia16_max16=[]
for i in inertia16:
    norm = i/max16
    #print(norm)
    inertia16_max16.append(norm)
print(inertia16_max16)

#normieten Inertia17

inertia17 = sorted(inertia17,reverse = True)
print(inertia17)
max17 = max(inertia17)
print(max17)
inertia17_max17=[]
for i in inertia17:
    norm = i/max17
    #print(norm)
    inertia17_max17.append(norm)
print(inertia17_max17)

#normierten Inertia 18

inertia18 = sorted(inertia18, reverse = True)
print(inertia18)
max18 = max(inertia18)
print(max18)
inertia18_max18=[]
for i in inertia18:
    norm = i/max18
    #print(norm)
    inertia18_max18.append(norm)
print(inertia18_max18)


# In[90]:


plt.figure(figsize =(15,11))
plt.plot(K16, inertia16_max16, 'x-')
plt.plot(K17, inertia17_max17, 'x-')
plt.plot(K18, inertia18_max18, 'x-')
plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('Der Elbow Kurv')
plt.legend(["gender', 'heart_disease', 'ever_married', 'Residence_type', 'avg_glucose_level","age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type","age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level"])
plt.show()


# In[91]:


#Normierten Inertia 19
inertia19 = sorted(inertia19,reverse = True)
print(inertia19)
max19 = max(inertia19)
print(max19)
inertia19_max19=[]
for i in inertia19:
    norm = i/max19
    #print(norm)
    inertia19_max19.append(norm)
print(inertia19_max19)

#Normieten Inertia 20

inertia20 = sorted(inertia20,reverse = True)
print(inertia20)
max20 = max(inertia20)
print(max20)
inertia20_max20=[]
for i in inertia20:
    norm = i/max20
    #print(norm)
    inertia20_max20.append(norm)
print(inertia20_max20)

#normierten Inertia 21

inertia21 = sorted(inertia21,reverse = True)
print(inertia21)
max21 = max(inertia21)
print(max21)
inertia21_max21=[]
for i in inertia21:
    norm = i/max21
    #print(norm)
    inertia21_max21.append(norm)
print(inertia21_max21)


# In[92]:


plt.figure(figsize =(15,11))
plt.plot(K19, inertia19_max19, 'x-')
plt.plot(K20, inertia20_max20, 'x-')
plt.plot(K21, inertia21_max21, 'x-')
plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('Der Elbow Kurv')
plt.legend(["age', 'hypertension', 'heart_disease', 'Residence_type', 'avg_glucose_level","age', 'hypertension', 'ever_married', 'Residence_type', 'avg_glucose_level","age', 'heart_disease', 'ever_married', 'Residence_type', 'avg_glucose_level","hypertension', 'heart_disease', 'ever_married', 'Residence_type', 'avg_glucose_level"])
plt.show()


# In[ ]:




