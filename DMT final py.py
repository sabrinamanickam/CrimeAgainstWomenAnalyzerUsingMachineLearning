#!/usr/bin/env python
# coding: utf-8

# # Crime against Women analysis
About Dataset It has state-wise and district level data on the various crimes committed against women between 2001 to 2014. Crimes that included are :

Rape
Kidnapping and Abduction
Dowry Deaths
Assault on women with intent to outrage her modesty
Insult to modesty of Women
Cruelty by Husband or his Relatives
Importation of Girl
# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import warnings
warnings.filterwarnings('ignore')


# ## Basic information of Data like Shape, data type, null values, Unique Characters

# In[4]:


data = pd.read_csv(r"C:\Users\LAASYA\Desktop\complete data.csv")
data.shape


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.isnull().sum()


# In[9]:


data.nunique()


# In[10]:


data = data.drop("S.No",axis=1)


# In[11]:


data.head()


# In[12]:


list(data["STATE/UT"].unique())


# ## Data Cleaning

# ### Data Cleaning on STATE/UT Column

In India we have only 36 States/UT but as per the data it has 72 States/UT, so we deep dive into find the errors and fix it, When we see the STATE/UT list we came to know that some of them are entered in upper Case and some in lower case. so the count has increased than actual. To clear this issue we converted all to a lower case and then also it shows 3 No. higher than the actual and while inspecting found that there is spacing issues and then fixed it to obtain the actual.

So, it is always good to compare it with the real data.
# In[13]:


list(data["STATE/UT"].unique())
len(list(data["STATE/UT"].unique()))


# In[14]:


data.replace({'A&N Islands':'A & N Islands','D&N Haveli':'D & N HAVELI','Delhi UT':'DELHI'},inplace=True)


# In[15]:


len(list(data["STATE/UT"].unique()))


# In[16]:


data["STATE/UT"]=data["STATE/UT"].str.casefold()


# In[17]:


len(list(data["STATE/UT"].unique()))


# In[18]:


data["DISTRICT"].value_counts()


# ### Data Cleaning on DISTRICT
Initially we converted all the entries to lower case, and then obtained the unique list of District, while inspecting found that it has entries like "total", "zz total", "total district(s)", "delhi ut total" this doesn't look like an district name so while investigating it seems to be an total value of that state on each crime. If it present in the data it will mess up the data by increasing the No. of cases in each category so, found the indexes of that total and removed from the dataset.
# In[19]:


data["DISTRICT"]=data["DISTRICT"].str.lower()


# In[20]:


data=data.drop(list(data[data["DISTRICT"]=="total"].index))


# In[21]:


len(list(data[data["DISTRICT"]=="zz total"].index))


# In[22]:


len(list(data[data["DISTRICT"]=="total district(s)"].index))


# In[23]:


len(list(data[data["DISTRICT"]=="delhi ut total"].index))


# In[24]:


data = data.drop(list(data[data["DISTRICT"]=="zz total"].index))


# In[25]:


data = data.drop(list(data[data["DISTRICT"]=="total district(s)"].index))


# In[26]:


data = data.drop(list(data[data["DISTRICT"]=="delhi ut total"].index))


# In[27]:


data["DISTRICT"].value_counts()


# In[28]:


data["DISTRICT"].unique()


# In[29]:


data.columns


# In[30]:


data["Total"]=data["Rape"]*7 + data["Kidnapping and Abduction"]*6 + data["Dowry Deaths"]*5 +data["Assault on women with intent to outrage her modesty"]*4 +data["Insult to modesty of Women"]*3 + data["Cruelty by Husband or his Relatives"]*2 + data["Importation of Girls"]*1


# In[31]:


data.head()


# ## Data Analysis

# ### Heat map

# In[32]:


fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(data.corr(), center = 0, cmap = "Reds")
ax.set_title("Crime against women Data")


# In[33]:


fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(data.corr(), center = 0, cmap = "BrBG", annot = True)


# ### Year Wise Analysis of Crime
The Below graph depicts the Year wise analysis of crime from 2001 to 2014. It seems like the crime rate increases rapidly as the year goes On.
# In[34]:


plt.figure(figsize=(12,8))
data.groupby("Year")["Total"].sum().plot.bar()


# #### Percentage Distribution of each Crime based on Year

# In[35]:


data.groupby("Year")["Rape"].sum().plot.pie(autopct='%1.0f%%')
plt.tight_layout()


# In[36]:


data.groupby("Year")["Kidnapping and Abduction"].sum().plot.pie(autopct='%1.0f%%')
plt.tight_layout()


# In[37]:


data.groupby("Year")["Dowry Deaths"].sum().plot.pie(autopct='%1.0f%%')
plt.tight_layout()


# In[38]:


data.groupby("Year")["Assault on women with intent to outrage her modesty"].sum().plot.pie(autopct='%1.0f%%')
plt.tight_layout()


# In[39]:


data.groupby("Year")["Insult to modesty of Women"].sum().plot.pie(autopct='%1.0f%%')
plt.tight_layout()


# In[40]:


data.groupby("Year")["Cruelty by Husband or his Relatives"].sum().plot.pie(autopct='%1.0f%%')
plt.tight_layout()


# In[41]:


data.groupby("Year")["Importation of Girls"].sum().plot.pie(autopct='%1.0f%%')
plt.tight_layout()


# #### Summary
The Below graph represents the year by year trend on each category of Crime.
# In[42]:


plt.figure(figsize=(20,12))
plt.subplot(2,4,1)
data.groupby("Year")["Rape"].sum().plot(title="Rape")
plt.subplot(2,4,2)
data.groupby("Year")["Kidnapping and Abduction"].sum().plot(title="Kidnapping and Abduction")
plt.subplot(2,4,3)
data.groupby("Year")["Dowry Deaths"].sum().plot(title="Dowry Deaths")
plt.subplot(2,4,6)
data.groupby("Year")["Assault on women with intent to outrage her modesty"].sum().plot(title="Assault on women with intent to outrage her modesty")
plt.subplot(2,4,5)
data.groupby("Year")["Insult to modesty of Women"].sum().plot(title="Insult to modesty of Women")
plt.subplot(2,4,4)
data.groupby("Year")["Cruelty by Husband or his Relatives"].sum().plot(title="Cruelty by Husband or his Relatives")
plt.subplot(2,4,7)
data.groupby("Year")["Importation of Girls"].sum().plot(title="Importation of Girls")
plt.subplot(2,4,8)
data.groupby("Year")["Total"].sum().plot(title="Total No. of Crimes")


# In[43]:


plt.figure(figsize=(15,7))
data.groupby("Year")["Rape"].sum().plot()
data.groupby("Year")["Kidnapping and Abduction"].sum().plot()
data.groupby("Year")["Dowry Deaths"].sum().plot(label="Dowry Deaths")
data.groupby("Year")["Assault on women with intent to outrage her modesty"].sum().plot()
data.groupby("Year")["Insult to modesty of Women"].sum().plot()
data.groupby("Year")["Cruelty by Husband or his Relatives"].sum().plot()
data.groupby("Year")["Importation of Girls"].sum().plot()
plt.legend()
plt.tight_layout()


# #### Yearwise Crime Rate on Different Categories

# In[44]:


crimes=['Rape','Kidnapping and Abduction','Dowry Deaths',
        'Assault on women with intent to outrage her modesty',
        'Insult to modesty of Women','Cruelty by Husband or his Relatives',
        'Importation of Girls']

data1=pd.DataFrame()
for i in crimes:
    data_crimes=data.groupby(['Year'])[i].sum()
    data1[i]=data_crimes
data1


# #### Percentage Contribution of Each Category of Crime

# In[45]:


a=[]
for i in crimes:
  a.append(data1[i].sum())
a.sort()
plt.figure(figsize=(10,15))
plt.pie(a,labels=crimes,autopct='%1.2f%%',colors=['black', 'gold', 'lightskyblue', 'lightcoral','lightpink','lightcyan','lightgreen'])
plt.tight_layout()


# In[46]:


data2 = data1.T
data2


# In[47]:


data2["crime"] = data2.index
data2


# ### State/UT wise Analysis of Crime

# In[48]:


plt.figure(figsize=(17,7))
data.groupby("STATE/UT")["Total"].sum().sort_values(ascending=False).plot.bar()


# In[49]:


crimes=['Rape','Kidnapping and Abduction','Dowry Deaths',
        'Assault on women with intent to outrage her modesty',
        'Insult to modesty of Women','Cruelty by Husband or his Relatives',
        'Importation of Girls']

data_state=pd.DataFrame()
for i in crimes:
    data_state_crimes=data.groupby(['STATE/UT'])[i].sum()
    data_state[i]=data_state_crimes
data_state["Total"]=data_state.sum(axis=1)
data_state = data_state.sort_values(by="Total",ascending=False)
data_state.reset_index()
data_state


# ### Top 3 States with Higher Number of Crimes

# In[50]:


data_state.head(3)


# 

# ## Label Encoding and Splitting data into Training and Testing data

# ### Label Encoding

# In[51]:


inputs = data.drop('Target_Label', axis = 'columns')
inputs= inputs.drop('Target', axis = 'columns')
target = data['Target_Label']


# In[52]:


from sklearn.preprocessing import LabelEncoder


# In[53]:


le_state = LabelEncoder()
le_district = LabelEncoder()
le_year = LabelEncoder()
le_targetlabel = LabelEncoder()


# In[54]:


inputs['STATE/UT_n'] = le_state.fit_transform(inputs['STATE/UT'])
inputs['DISTRICT_n'] = le_district.fit_transform(inputs['DISTRICT'])
inputs['Year_n'] = le_year.fit_transform(inputs['Year'])
target = le_targetlabel.fit_transform(target)


# In[55]:


inputs.head()


# In[56]:


target


# In[57]:


inputs_n = inputs.drop(['STATE/UT', 'DISTRICT' , 'Year', 'Total'], axis = 'columns')


# In[58]:


inputs_n.head()


# ### Splitting

# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


X_train, X_test, Y_train, Y_test = train_test_split(inputs_n, target ,random_state= 101, stratify= target, train_size = 0.7)
X_train.shape , X_test.shape


# In[61]:


accuracy = []


# ## Classification

# ### Decision Tree

# In[62]:


from sklearn.tree import DecisionTreeClassifier


# In[63]:


from sklearn.metrics import confusion_matrix


# In[64]:


dt_model=DecisionTreeClassifier(random_state=10)


# In[65]:


dt_model.fit(X_train,Y_train)


# In[66]:


dt_model.score(X_train,Y_train)


# In[67]:


dt_model.score(X_test,Y_test)


# In[68]:


dt_accuracy = dt_model.score(X_test,Y_test)


# In[69]:


accuracy.append(dt_accuracy*100)


# In[70]:


result=dt_model.predict(X_test)


# In[71]:


result


# In[72]:


data_dt_cls=pd.DataFrame({'Actual':Y_test, 'Predicted':result})
data_dt_cls


# In[73]:


dt_model.predict_proba(X_test)


# #### Confusion Matrix

# In[74]:


from sklearn import metrics


# In[75]:



print(confusion_matrix(Y_test, result))


# In[76]:


mat = confusion_matrix(result, Y_test)
names = np.unique(result)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')


# #### Precision, Recall and F1 Score

# In[77]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_score,f1_score
from sklearn.metrics import recall_score


# In[78]:


precision = precision_score(Y_test, result,average='weighted')
recall = recall_score(Y_test, result,average='weighted')
score = f1_score(Y_test, result, average='weighted')
 
print('Precision: ',precision)
print('Recall: ',recall)
print('F1_Score: ',score)


# #### Mean Absolute Error, MSE, RMSE

# In[79]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, result))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, result))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, result)))


# #### Plot a tree

# In[80]:


from sklearn import tree
decision_tree=tree.export_graphviz(dt_model,out_file='tree.dot',feature_names=X_train.columns,max_depth=10,filled=True)


# In[81]:


dt_model.predict([[1,2,3,4,5,6,7,9,8,0]])


# ### Random Forest Classifier

# In[82]:


from sklearn.ensemble import RandomForestClassifier


# In[83]:


classifier_rf = RandomForestClassifier(n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)


# In[84]:


classifier_rf.fit(X_train, Y_train)


# In[85]:


classifier_rf.oob_score_


# In[86]:


rf = RandomForestClassifier(n_jobs=-1)


# In[87]:


from sklearn.model_selection import GridSearchCV


# In[88]:


params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200]
}


# In[89]:


grid_search = GridSearchCV(estimator=rf,
param_grid=params,
cv = 4,
n_jobs=-1, verbose=1, scoring="accuracy")


# In[90]:


grid_search.fit(X_train,Y_train)


# In[91]:


grid_search.best_score_


# In[92]:


rf_best = grid_search.best_estimator_
rf_best


# In[93]:


from sklearn.tree import plot_tree
plt.figure(figsize=(80,40))
plot_tree(rf_best.estimators_[5], feature_names = inputs_n.columns,class_names=['LOW', 'HIGH', 'MEDIUM'],filled=True);


# In[94]:


from sklearn.tree import plot_tree
plt.figure(figsize=(80,40))
plot_tree(rf_best.estimators_[5], feature_names = X_train.columns,class_names=['LOW', 'HIGH', 'MEDIUM'],filled=True);


# In[95]:


from sklearn.tree import plot_tree
plt.figure(figsize=(80,40))
plot_tree(rf_best.estimators_[7], feature_names = X_test.columns,class_names=['LOW', 'HIGH', 'MEDIUM'],filled=True);


# In[96]:


rf_best.feature_importances_


# In[97]:


imp_df = pd.DataFrame({
    "Varname": X_train.columns,
    "Imp": rf_best.feature_importances_
})


# In[98]:


imp_df.sort_values(by="Imp", ascending=False)


# In[99]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[100]:


inputs_n.head()


# In[101]:


rf.fit(X_train,Y_train)


# In[102]:


rf.score(X_test, Y_test)


# In[103]:


rf_accuracy = rf.score(X_test, Y_test)


# In[104]:


accuracy.append(rf_accuracy*100)


# In[105]:


result = rf.predict(X_test)


# In[106]:


result


# In[107]:


data_rf_cls = pd.DataFrame({'Actual': Y_test, 'Predicted': result})
data_rf_cls


# In[108]:


rf.predict_proba(X_test)


# #### Confusion Matrix

# In[109]:


print(confusion_matrix(Y_test, result))


# In[110]:


mat = confusion_matrix(result, Y_test)
names = np.unique(result)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')


# #### Precision, Recall and F1 Score

# In[111]:


precision = precision_score(Y_test, result,average='weighted')
recall = recall_score(Y_test, result,average='weighted')
score = f1_score(Y_test, result, average='weighted')
 
print('Precision: ',precision)
print('Recall: ',recall)
print('F1_Score: ',score)


# #### Mean Absolute Error, MSE, RMSE

# In[112]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, result))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, result))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, result)))


# In[113]:


rf.predict([[1,2,3,4,5,6,7,9,8,0]])


# ## Naive Bayes 

# In[114]:


from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()


# In[115]:


nb_model.fit(X_train, Y_train)


# In[116]:


nb_model.score(X_train,Y_train)


# In[117]:


nb_model.score(X_test,Y_test)


# In[118]:


nb_accuracy = nb_model.score(X_test,Y_test)


# In[119]:


accuracy.append(nb_accuracy*100)


# In[120]:


result=nb_model.predict(X_test)


# In[121]:


result


# In[122]:


data_nb_cls=pd.DataFrame({'Actual':Y_test, 'Predicted':result})
data_nb_cls


# In[123]:


nb_model.predict_proba(X_test)


# #### Confusion Matrix

# In[124]:


print(confusion_matrix(Y_test, result))


# In[125]:


mat = confusion_matrix(result, Y_test)
names = np.unique(result)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')


# #### Precision, Recall and F1 Score

# In[126]:


precision = precision_score(Y_test, result,average='weighted')
recall = recall_score(Y_test, result,average='weighted')
score = f1_score(Y_test, result, average='weighted')
 
print('Precision: ',precision)
print('Recall: ',recall)
print('F1_Score: ',score)


# #### Mean Absolute Error, MSE, RMSE

# In[127]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, result))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, result))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, result)))


# #### Plot a tree

# In[128]:


nb_model.predict([[1,2,3,4,5,6,7,9,8,0]])


# ## Support Vector machine

# In[129]:


from sklearn.svm import SVC
SVM_model = SVC()


# In[130]:


SVM_model.fit(X_train,Y_train)


# In[131]:


SVM_model.score(X_train,Y_train)


# In[132]:


SVM_model.score(X_test,Y_test)


# In[133]:


SVM_accuracy = SVM_model.score(X_test,Y_test)


# In[134]:


accuracy.append(SVM_accuracy*100)


# In[135]:


result=SVM_model.predict(X_test)


# In[136]:


result


# In[137]:


data_svm_cls=pd.DataFrame({'Actual':Y_test, 'Predicted':result})
data_svm_cls


# #### Confusion matrix

# In[138]:


print(confusion_matrix(Y_test, result))


# In[139]:


mat = confusion_matrix(result, Y_test)
names = np.unique(result)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')


# #### Precision, Recall and F1 Score

# In[140]:


precision = precision_score(Y_test, result,average='weighted')
recall = recall_score(Y_test, result,average='weighted')
score = f1_score(Y_test, result, average='weighted')
 
print('Precision: ',precision)
print('Recall: ',recall)
print('F1_Score: ',score)


# #### Mean Absolute Error, MSE, RMSE

# In[141]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, result))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, result))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, result)))


# In[142]:


SVM_model.predict([[1,2,3,4,5,6,7,9,8,0]])


# ### Classification accuracy plot

# In[143]:


accuracy


# In[144]:


models = ("DT" , "RFC" , "NB" , "SVM")


# In[145]:


plt.bar(models , accuracy , width = 0.8, color = ['lightskyblue', 'lightcoral','lightpink','cyan'])
plt.title("Accuracy of various algorithms")

plt.xlabel("Various Algorithms")
plt.ylabel("Accuracy")
plt.ylim(96, 100.0)

plt.show()


# ## Regression

# In[146]:


r_accuracy = []


# ### Decision Tree 

# In[147]:


from sklearn.tree import DecisionTreeRegressor


# In[148]:


dt_regressor = DecisionTreeRegressor(random_state=0)


# In[149]:


dt_regressor.fit(X_train,Y_train)


# In[150]:


dt_regressor.predict(X_test)


# In[151]:


dt_regressor.score(X_train,Y_train)


# In[152]:


dt_regressor.score(X_test,Y_test)


# In[153]:


dt_accuracy = dt_regressor.score(X_test,Y_test)


# In[154]:


r_accuracy.append(dt_accuracy * 100)


# In[155]:


result_dt_reg=dt_regressor.predict(X_test)


# In[156]:


result_dt_reg


# In[157]:


data_dt_reg=pd.DataFrame({'Actual':Y_test, 'Predicted':result_dt_reg})
data_dt_reg


# In[158]:


precision = precision_score(Y_test, result_dt_reg,average='weighted')
recall = recall_score(Y_test, result_dt_reg,average='weighted')
score = f1_score(Y_test, result_dt_reg, average='weighted')
print('Precision: ',precision)
print('Recall: ',recall)
print('F1_Score: ',score)


# #### Plot a tree

# In[159]:


from sklearn import tree
dt_reg_tree =tree.export_graphviz(dt_regressor,out_file='dt_reg_tree.dot',feature_names=X_train.columns,max_depth=10,filled=True)


# In[160]:


dt_regressor.predict([[1,2,3,4,5,6,7,9,8,0]])


# ### Random Forest 

# In[161]:


from sklearn.ensemble import RandomForestRegressor


# In[162]:


rf_regressor = RandomForestRegressor(random_state=0)


# In[163]:


rf_regressor.fit(X_train,Y_train)


# In[164]:


rf_regressor.predict(X_test)


# In[165]:


rf_regressor.score(X_train,Y_train)


# In[166]:


rf_regressor.score(X_test,Y_test)


# In[167]:


rf_accuracy = rf_regressor.score(X_test,Y_test)


# In[168]:


r_accuracy.append(rf_accuracy * 100)


# In[169]:


result_rf_reg=rf_regressor.predict(X_test)


# In[170]:


result_rf_reg


# In[171]:


data_rf_reg=pd.DataFrame({'Actual':Y_test, 'Predicted':result_rf_reg})
data_rf_reg


# ### Linear Regression

# In[172]:


from sklearn.linear_model import LinearRegression


# In[173]:


l_reg = LinearRegression()


# In[174]:


l_reg.fit(X_train, Y_train)


# In[175]:


l_reg.predict(X_test)


# In[176]:


l_reg.score(X_train,Y_train)


# In[177]:


l_reg.score(X_test,Y_test)


# In[178]:


l_reg_accuracy = l_reg.score(X_test,Y_test)


# In[179]:


r_accuracy.append(l_reg_accuracy * 100)


# In[180]:


result_ln_reg=l_reg.predict(X_test)


# In[181]:


result_ln_reg


# In[182]:


data_ln=pd.DataFrame({'Actual':Y_test, 'Predicted':result_ln_reg})
data_ln


# In[183]:


l_reg.predict([[1,2,3,4,5,6,7,9,8,0]])


# ### Regression Accuracy plot

# In[184]:


r_accuracy


# In[185]:


r_models = ("DT", "RF", "Linear")


# In[186]:


plt.bar(r_models , r_accuracy , width = 0.8, color = ['lightskyblue', 'lightcoral','lightpink'])
plt.title("Accuracy of various algorithms")

plt.xlabel("Various Algorithms")
plt.ylabel("Accuracy")
plt.ylim(0,70)

plt.show()


# ## Clustering

# ### BIRCH

# In[187]:



from sklearn.datasets import make_blobs
from sklearn.cluster import Birch


model=Birch(branching_factor=50,n_clusters=None, threshold=1.5)

model.fit(inputs_n)


# In[188]:


pred = model.predict(inputs_n)

pred


plt.scatter(inputs_n['Rape'],pred,cmap = 'rainbow', alpha = 0.7, edgecolors = 'b')


# In[189]:



from sklearn.datasets import make_blobs
from sklearn.cluster import Birch
data, clusters = make_blobs(n_samples = 1000, centers = 12, cluster_std = 0.50, random_state = 0)
data.shape
model = Birch(branching_factor = 50, n_clusters = None, threshold = 1.5)
model.fit(data)
pred = model.predict(data)
plt.scatter(data[:, 0], data[:, 1], c = pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




