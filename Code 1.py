#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


data = pd.read_csv("datahealth.csv")
data


# In[11]:


from itertools import combinations as com
import sklearn.cluster as cluster


# In[12]:


list_of_Tupel = data.columns
list_of_Tupel = list_of_Tupel.drop("stroke")
list_of_Tupel = list_of_Tupel.drop("id")
print(list_of_Tupel)


# In[13]:


#Mögliche Tupel auflisten
anzahl = 0
for tupel in com(list_of_Tupel,2):
   anzahl+=1
   print(anzahl)
   print(tupel)


# In[14]:


del data['stroke']
del data['id']
data


# In[15]:


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

#work_type
scaler.fit(data[['work_type']])
data['work_type'] = scaler.transform(data[['work_type']])

#Residence_type
scaler.fit(data[['Residence_type']])
data['Residence_type'] = scaler.transform(data[['Residence_type']])

#avg_glucose_level
scaler.fit(data[['avg_glucose_level']])
data['avg_glucose_level'] = scaler.transform(data[['avg_glucose_level']])

#bmi
scaler.fit(data[['bmi']])
data['bmi'] = scaler.transform(data[['bmi']])

#smoking_status
scaler.fit(data[['smoking_status']])
data['smoking_status'] = scaler.transform(data[['smoking_status']])

data


# In[16]:


# Daten auf 1 normieren
#Mit einer While SChleife
#i=0 
#while 1< 11:
    #tupel = list_of_tupel[i]
   # scaler= MInMAxScaler()
    #scaler.fit(data[[tupel]])
    #data[tupel] = scaler.transform(data[[tupel]])
   # i+=1
#data


# In[17]:


#tupel darstellen
tupel1=data[['gender', 'age']]

tupel2=data[['gender', 'hypertension']]

tupel3=data[['gender', 'heart_disease']]

tupel4=data[['gender', 'ever_married']]

tupel5=data[['gender', 'work_type']]

tupel6=data[['gender', 'Residence_type']]

tupel7=data[['gender', 'avg_glucose_level']]

tupel8=data[['gender', 'bmi']]

tupel9=data[['gender', 'smoking_status']]

tupel10=data[['age', 'hypertension']]

tupel11=data[['age', 'heart_disease']]

tupel12=data[['age', 'ever_married']]

tupel13=data[['age', 'work_type']]

tupel14=data[['age', 'Residence_type']]

tupel15=data[['age', 'avg_glucose_level']]

tupel16=data[['age', 'bmi']]

tupel17=data[['age', 'smoking_status']]

tupel18=data[['hypertension', 'heart_disease']]

tupel19=data[['hypertension', 'ever_married']]

tupel20=data[['hypertension', 'work_type']]

tupel21=data[['hypertension', 'Residence_type']]

tupel22=data[['hypertension', 'avg_glucose_level']]

tupel23=data[['hypertension', 'bmi']]

tupel24=data[['hypertension', 'smoking_status']]

tupel25=data[['heart_disease', 'ever_married']]

tupel26=data[['heart_disease', 'work_type']]

tupel27=data[['heart_disease', 'Residence_type']]

tupel28=data[['heart_disease', 'avg_glucose_level']]

tupel29=data[['heart_disease', 'bmi']]

tupel30=data[['heart_disease', 'smoking_status']]

tupel31=data[['ever_married', 'work_type']]

tupel32=data[['ever_married', 'Residence_type']]

tupel33=data[['ever_married', 'avg_glucose_level']]

tupel34=data[['ever_married', 'bmi']]

tupel35=data[['ever_married', 'smoking_status']]

tupel36=data[['work_type', 'Residence_type']]

tupel37=data[['work_type', 'avg_glucose_level']]

tupel38=data[['work_type', 'bmi']]

tupel39=data[['work_type', 'smoking_status']]

tupel40=data[['Residence_type', 'avg_glucose_level']]

tupel41=data[['Residence_type', 'bmi']]

tupel42=data[['Residence_type', 'smoking_status']]

tupel43=data[['avg_glucose_level', 'bmi']]

tupel44=data[['avg_glucose_level', 'smoking_status']]

tupel45=data[['bmi', 'smoking_status']]


# In[50]:


T1 = range(1,10)
inertia1 = []
for t1 in T1:
    kmeans = cluster.KMeans(n_clusters = t1, init = "k-means++")
    kmeans = kmeans.fit(tupel1)
    inertia_iter1 = kmeans.inertia_
    inertia1.append(inertia_iter1)


# In[19]:


T2 = range(1,10)
inertia2 = []
for t2 in T2:
    kmeans = cluster.KMeans(n_clusters = t2, init = "k-means++")
    kmeans = kmeans.fit(tupel2)
    inertia_iter2 = kmeans.inertia_
    inertia2.append(inertia_iter2)


# In[20]:


T3 = range(1,10)
inertia3 = []
for t3 in T3:
    kmeans = cluster.KMeans(n_clusters = t3, init = "k-means++")
    kmeans = kmeans.fit(tupel3)
    inertia_iter3 = kmeans.inertia_
    inertia3.append(inertia_iter3)


# In[21]:


T4 = range(1,10)
inertia4 = []
for t4 in T4:
    kmeans = cluster.KMeans(n_clusters = t4, init = "k-means++")
    kmeans = kmeans.fit(tupel4)
    inertia_iter4 = kmeans.inertia_
    inertia4.append(inertia_iter4)


# In[22]:


T5 = range(1,10)
inertia5 = []
for t5 in T5:
    kmeans = cluster.KMeans(n_clusters = t5, init = "k-means++")
    kmeans = kmeans.fit(tupel5)
    inertia_iter5 = kmeans.inertia_
    inertia5.append(inertia_iter5)


# In[23]:


T6 = range(1,10)
inertia6 = []
for t6 in T6:
    kmeans = cluster.KMeans(n_clusters = t6, init = "k-means++")
    kmeans = kmeans.fit(tupel6)
    inertia_iter6 = kmeans.inertia_
    inertia6.append(inertia_iter6)


# In[24]:


T7 = range(1,10)
inertia7 = []
for t7 in T7 :
    kmeans = cluster.KMeans(n_clusters = t7, init = "k-means++")
    kmeans = kmeans.fit(tupel7)
    inertia_iter7 = kmeans.inertia_
    inertia7.append(inertia_iter7)


# In[25]:


T8 = range(1,10)
inertia8 = []
for t8 in T8:
    kmeans = cluster.KMeans(n_clusters = t8, init = "k-means++")
    kmeans = kmeans.fit(tupel8)
    inertia_iter8 = kmeans.inertia_
    inertia8.append(inertia_iter8)


# In[26]:


T9 = range(1,10)
inertia9 = []
for t9 in T9:
    kmeans = cluster.KMeans(n_clusters = t9, init = "k-means++")
    kmeans = kmeans.fit(tupel9)
    inertia_iter9 = kmeans.inertia_
    inertia9.append(inertia_iter9)


# In[27]:


T10 = range(1,10)
inertia10 = []
for t10 in T10:
    kmeans = cluster.KMeans(n_clusters = t10, init = "k-means++")
    kmeans = kmeans.fit(tupel10)
    inertia_iter10 = kmeans.inertia_
    inertia10.append(inertia_iter10)


# In[28]:


T11 = range(1,10)
inertia11 = []
for t11 in T11:
    kmeans = cluster.KMeans(n_clusters = t11, init = "k-means++")
    kmeans = kmeans.fit(tupel11)
    inertia_iter11 = kmeans.inertia_
    inertia11.append(inertia_iter11)


# In[29]:


T12 = range(1,10)
inertia12 = []
for t12 in T12:
    kmeans = cluster.KMeans(n_clusters = t12, init = "k-means++")
    kmeans = kmeans.fit(tupel12)
    inertia_iter12 = kmeans.inertia_
    inertia12.append(inertia_iter12)


# In[30]:


T13 = range(1,10)
inertia13 = []
for t13 in T13:
    kmeans = cluster.KMeans(n_clusters = t13, init = "k-means++")
    kmeans = kmeans.fit(tupel13)
    inertia_iter13 = kmeans.inertia_
    inertia13.append(inertia_iter13)


# In[31]:


T14 = range(1,10)
inertia14 = []
for t14 in T14:
    kmeans = cluster.KMeans(n_clusters = t14, init = "k-means++")
    kmeans = kmeans.fit(tupel14)
    inertia_iter14 = kmeans.inertia_
    inertia14.append(inertia_iter14)


# In[32]:


T15 = range(1,10)
inertia15 = []
for t15 in T15:
    kmeans = cluster.KMeans(n_clusters = t15, init = "k-means++")
    kmeans = kmeans.fit(tupel15)
    inertia_iter15 = kmeans.inertia_
    inertia15.append(inertia_iter15)


# In[33]:


T16 = range(1,10)
inertia16 = []
for t16 in T16:
    kmeans = cluster.KMeans(n_clusters = t16, init = "k-means++")
    kmeans = kmeans.fit(tupel16)
    inertia_iter16 = kmeans.inertia_
    inertia16.append(inertia_iter16)


# In[34]:


T17 = range(1,10)
inertia17 = []
for t17 in T17 :
    kmeans = cluster.KMeans(n_clusters = t17, init = "k-means++")
    kmeans = kmeans.fit(tupel17)
    inertia_iter17 = kmeans.inertia_
    inertia17.append(inertia_iter17)


# In[35]:


T18 = range(1,10)
inertia18 = []
for t18 in T18:
    kmeans = cluster.KMeans(n_clusters = t18, init = "k-means++")
    kmeans = kmeans.fit(tupel18)
    inertia_iter18 = kmeans.inertia_
    inertia18.append(inertia_iter18)


# In[36]:


T19 = range(1,10)
inertia19 = []
for t19 in T19:
    kmeans = cluster.KMeans(n_clusters = t19, init = "k-means++")
    kmeans = kmeans.fit(tupel19)
    inertia_iter19 = kmeans.inertia_
    inertia19.append(inertia_iter19)


# In[37]:


T20 = range(1,10)
inertia20 = []
for t20 in T20:
    kmeans = cluster.KMeans(n_clusters = t20, init = "k-means++")
    kmeans = kmeans.fit(tupel20)
    inertia_iter20 = kmeans.inertia_
    inertia20.append(inertia_iter20)


# In[38]:


T21 = range(1,10)
inertia21 = []
for t21 in T21:
    kmeans = cluster.KMeans(n_clusters = t21, init = "k-means++")
    kmeans = kmeans.fit(tupel21)
    inertia_iter21 = kmeans.inertia_
    inertia21.append(inertia_iter21)


# In[39]:


T22 = range(1,10)
inertia22 = []
for t22 in T22:
    kmeans = cluster.KMeans(n_clusters = t22, init = "k-means++")
    kmeans = kmeans.fit(tupel22)
    inertia_iter22 = kmeans.inertia_
    inertia22.append(inertia_iter22)


# In[40]:


T23 = range(1,10)
inertia23 = []
for t23 in T23:
    kmeans = cluster.KMeans(n_clusters = t23, init = "k-means++")
    kmeans = kmeans.fit(tupel23)
    inertia_iter23 = kmeans.inertia_
    inertia23.append(inertia_iter23)


# In[41]:


T25 = range(1,10)
inertia25 = []
for t25 in T25:
    kmeans = cluster.KMeans(n_clusters = t25, init = "k-means++")
    kmeans = kmeans.fit(tupel25)
    inertia_iter25 = kmeans.inertia_
    inertia25.append(inertia_iter25)


# In[42]:


T26 = range(1,10)
inertia26 = []
for t26 in T26:
    kmeans = cluster.KMeans(n_clusters = t26, init = "k-means++")
    kmeans = kmeans.fit(tupel26)
    inertia_iter26 = kmeans.inertia_
    inertia26.append(inertia_iter26)


# In[43]:


T27 = range(1,10)
inertia27 = []
for t27 in T27 :
    kmeans = cluster.KMeans(n_clusters = t27, init = "k-means++")
    kmeans = kmeans.fit(tupel27)
    inertia_iter27 = kmeans.inertia_
    inertia27.append(inertia_iter27)


# In[44]:


T28 = range(1,10)
inertia28 = []
for t28 in T28:
    kmeans = cluster.KMeans(n_clusters = t28, init = "k-means++")
    kmeans = kmeans.fit(tupel8)
    inertia_iter28 = kmeans.inertia_
    inertia28.append(inertia_iter28)


# In[45]:


T29 = range(1,10)
inertia29 = []
for t29 in T29:
    kmeans = cluster.KMeans(n_clusters = t29, init = "k-means++")
    kmeans = kmeans.fit(tupel29)
    inertia_iter29 = kmeans.inertia_
    inertia29.append(inertia_iter29)


# In[46]:


T24 = range(1,10)
inertia24 = []
for t24 in T24:
    kmeans = cluster.KMeans(n_clusters = t24, init = "k-means++")
    kmeans = kmeans.fit(tupel24)
    inertia_iter24 = kmeans.inertia_
    inertia24.append(inertia_iter24)


# In[47]:


T30 = range(1,10)
inertia30 = []
for t30 in T30:
    kmeans = cluster.KMeans(n_clusters = t30, init = "k-means++")
    kmeans = kmeans.fit(tupel30)
    inertia_iter30 = kmeans.inertia_
    inertia30.append(inertia_iter30)


# In[48]:


T31 = range(1,10)
inertia31 = []
for t31 in T31:
    kmeans = cluster.KMeans(n_clusters = t31, init = "k-means++")
    kmeans = kmeans.fit(tupel31)
    inertia_iter31 = kmeans.inertia_
    inertia.append(inertia_iter31)


# In[ ]:


T32 = range(1,10)
inertia32 = []
for t32 in T32:
    kmeans = cluster.KMeans(n_clusters = t32, init = "k-means++")
    kmeans = kmeans.fit(tupel32)
    inertia_iter32 = kmeans.inertia_
    inertia32.append(inertia_iter32)


# In[64]:


T33 = range(1,10)
inertia33 = []
for t33 in T33:
    kmeans = cluster.KMeans(n_clusters = t33, init = "k-means++")
    kmeans = kmeans.fit(tupel33)
    inertia_iter33 = kmeans.inertia_
    inertia33.append(inertia_iter33)


# In[65]:


T34 = range(1,10)
inertia34 = []
for t34 in T34:
    kmeans = cluster.KMeans(n_clusters = t34, init = "k-means++")
    kmeans = kmeans.fit(tupel34)
    inertia_iter34 = kmeans.inertia_
    inertia34.append(inertia_iter34)


# In[66]:


T35 = range(1,10)
inertia35 = []
for t35 in T35:
    kmeans = cluster.KMeans(n_clusters = t35, init = "k-means++")
    kmeans = kmeans.fit(tupel35)
    inertia_iter35 = kmeans.inertia_
    inertia35.append(inertia_iter35)


# In[67]:


T36 = range(1,10)
inertia36 = []
for t36 in T36:
    kmeans = cluster.KMeans(n_clusters = t36, init = "k-means++")
    kmeans = kmeans.fit(tupel36)
    inertia_iter36 = kmeans.inertia_
    inertia36.append(inertia_iter36)


# In[68]:


T37 = range(1,10)
inertia37 = []
for t37 in T37 :
    kmeans = cluster.KMeans(n_clusters = t37, init = "k-means++")
    kmeans = kmeans.fit(tupel37)
    inertia_iter37 = kmeans.inertia_
    inertia37.append(inertia_iter37)


# In[69]:


T38 = range(1,10)
inertia38 = []
for t38 in T38:
    kmeans = cluster.KMeans(n_clusters = t38, init = "k-means++")
    kmeans = kmeans.fit(tupel38)
    inertia_iter38 = kmeans.inertia_
    inertia38.append(inertia_iter38)


# In[70]:


T39 = range(1,10)
inertia39 = []
for t39 in T39:
    kmeans = cluster.KMeans(n_clusters = t39, init = "k-means++")
    kmeans = kmeans.fit(tupel39)
    inertia_iter39 = kmeans.inertia_
    inertia39.append(inertia_iter39)


# In[71]:


T40= range(1,10)
inertia40 = []
for t40 in T40:
    kmeans = cluster.KMeans(n_clusters = t40, init = "k-means++")
    kmeans = kmeans.fit(tupel40)
    inertia_iter40 = kmeans.inertia_
    inertia40.append(inertia_iter40)


# In[72]:


T41= range(1,10)
inertia41 = []
for t41 in T41:
    kmeans = cluster.KMeans(n_clusters = t41, init = "k-means++")
    kmeans = kmeans.fit(tupel41)
    inertia_iter41 = kmeans.inertia_
    inertia41.append(inertia_iter41)


# In[73]:


T42 = range(1,10)
inertia42 = []
for t42 in T42:
    kmeans = cluster.KMeans(n_clusters = t42, init = "k-means++")
    kmeans = kmeans.fit(tupel42)
    inertia_iter42 = kmeans.inertia_
    inertia42.append(inertia_iter42)


# In[74]:


T43 = range(1,10)
inertia43 = []
for t43 in T43:
    kmeans = cluster.KMeans(n_clusters = t43, init = "k-means++")
    kmeans = kmeans.fit(tupel43)
    inertia_iter43 = kmeans.inertia_
    inertia43.append(inertia_iter43)


# In[75]:


T44 = range(1,10)
inertia44 = []
for t44 in T44:
    kmeans = cluster.KMeans(n_clusters = t44, init = "k-means++")
    kmeans = kmeans.fit(tupel44)
    inertia_iter44 = kmeans.inertia_
    inertia44.append(inertia_iter44)


# In[76]:


T45 = range(1,10)
inertia45 = []
for t45 in T45:
    kmeans = cluster.KMeans(n_clusters = t45, init = "k-means++")
    kmeans = kmeans.fit(tupel45)
    inertia_iter45 = kmeans.inertia_
    inertia45.append(inertia_iter45)


# In[51]:



plt.figure(figsize =(15,11))
plt.plot(T1, inertia1, 'x-')
plt.plot(T2, inertia2, 'x-')
plt.plot(T3, inertia3, 'x-')
plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('This Elbow Kurv zeigt die optimiert K')
plt.legend(["genderVsage","genderVsHpertension", "genderVsHeart_deseases"])
plt.show()


# In[52]:



plt.figure(figsize =(15,11))
plt.plot(T4, inertia4, 'x-')
plt.plot(T5, inertia5, 'x-')
plt.plot(T6, inertia6, 'x-')
plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('This Elbow Kurv zeigt die optimierte K')
plt.legend(["genderVsever_married","GenderVsWorkType", "genderVsresidence"])
plt.show()


# In[53]:


plt.figure(figsize =(15,11))
plt.plot(T7, inertia7, 'x-')
plt.plot(T8, inertia8, 'x-')
plt.plot(T9, inertia9, 'x-')

plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('This Elbow Kurv zeigt die optimierte K')
plt.legend(["genderVsGlucose","GenderVsBMI", "genderVsSmokingStatus","genderVsHypertension"])
plt.show()


# In[54]:


plt.figure(figsize =(15,11))
plt.plot(T10, inertia10, 'x-')
plt.plot(T11, inertia11, 'x-')
plt.plot(T12, inertia12, 'x-')
plt.plot(T13, inertia13, 'x-')

plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('This Elbow Kurv zeigt die optimierte K')
plt.legend(["ageVshypertension","ageVsheart_disease", "ageVsever_married","ageVswork_type"])
plt.show()


# In[55]:


plt.figure(figsize =(15,11))
plt.plot(T14, inertia14, 'x-')
plt.plot(T15, inertia15, 'x-')
plt.plot(T16, inertia16, 'x-')
plt.plot(T17, inertia17, 'x-')

plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('This Elbow Kurv zeigt die optimierte K')
plt.legend(["ageVsResidence_type","ageVsGlucose", "ageVsBmi","ageVsSmoking_status"])
plt.show()


# In[56]:


plt.figure(figsize =(15,11))
plt.plot(T18, inertia18, 'x-')
plt.plot(T19, inertia19, 'x-')
plt.plot(T20, inertia20, 'x-')


plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('This Elbow Kurv zeigt die optimierte K')
plt.legend(["hypertensionVsHeart_disease","hypertensionVsEver_married", "HypertensionVswork_type"])
plt.show()


# In[57]:


plt.figure(figsize =(15,11))
plt.plot(T21, inertia21, 'x-')
plt.plot(T22, inertia22, 'x-')
plt.plot(T23, inertia23, 'x-')
plt.plot(T24, inertia24, 'x-')



plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('This Elbow Kurv zeigt die optimierte K')
plt.legend(["hypertensionVsResidence_type","hypertensionVsavg_glucose_level", "hypertensionVsBmi","hypertensionVssmoking_status"])
plt.show()


# In[58]:


plt.figure(figsize =(15,11))
plt.plot(T25, inertia25, 'x-')
plt.plot(T26, inertia26, 'x-')
plt.plot(T27, inertia27, 'x-')
plt.plot(T28, inertia28, 'x-')

plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('This Elbow Kurv zeigt die optimierte K')
plt.legend(["heart_diseaseVSever_married","heart_diseaseVswork_type", "heart_diseaseVsResidence_type","heart_diseaseVsavg_glucose_level"])
plt.show()


# In[59]:


plt.figure(figsize =(15,11))
plt.plot(T29, inertia29, 'x-')
plt.plot(T30, inertia30, 'x-')
plt.plot(T31, inertia31, 'x-')
plt.plot(T32, inertia32, 'x-')

plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('This Elbogen Kurv zeigt die optimierte K')
plt.legend(["heart_diseaseVsBmi","heart_diseaseVsSmoking_status", "ever_marriedVswork_type","ever_marriedVsResidence_type"])
plt.show()


# In[60]:


plt.figure(figsize =(15,11))
plt.plot(T33, inertia33, 'x-')
plt.plot(T34, inertia34, 'x-')
plt.plot(T35, inertia35, 'x-')
plt.plot(T36, inertia36, 'x-')

plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('This Elbow Kurv zeigt die optimierte K')
plt.legend(["ever_marriedVsavg_glucose_level","ever_marriedVsbmi", "ever_marriedVssmoking_status","work_typeVsResidence_type"])
plt.show()


# In[61]:


plt.figure(figsize =(15,11))
plt.plot(T37, inertia37, 'x-')
plt.plot(T38, inertia38, 'x-')
plt.plot(T39, inertia39, 'x-')
plt.plot(T40, inertia40, 'x-')

plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('This Elbow Kurv zeigt die optimierte K')
plt.legend(["work_typeVsavg_glucose_level","work_typeVsbmi", "work_typeVssmoking_status","Residence_typeVsavg_glucose_level"])
plt.show()


# In[62]:


plt.figure(figsize =(15,11))
plt.plot(T44, inertia44, 'x-')
plt.plot(T45, inertia45, 'x-')

plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('This Elbow Kurv zeigt die optimierte K')
plt.legend(["avg_glucose_level', 'smoking_status","bmi', 'smoking_status"])
plt.show()


# In[63]:


plt.figure(figsize =(15,11))
plt.plot(T41, inertia41, 'x-')
plt.plot(T42, inertia42, 'x-')
plt.plot(T43, inertia43, 'x-')


plt.xlabel('K')
plt.ylabel('Inertia Werte')
plt.title('This Elbow Kurv zeigt die optimierte K')
plt.legend(["Residence_typeVsBmi", "Residence_typeVsSmoking_status", "avg_glucose_levelVsBmi"])
plt.show()


# In[ ]:




