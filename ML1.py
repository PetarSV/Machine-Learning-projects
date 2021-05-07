# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:23:12 2020

@author: Petar 
"""


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

#%% Loading data


data = []


with open('mouse_train.txt', "r") as f:
    for line in f:
        if line[0] == '-':
            data.append(line)
        else:
            linija = line[1:-2] # Parenthesis removal
            tmp_list = linija.split(',')    # Comma removal
            tmp_list2 = [int(x) for x in tmp_list]  # string to int
            data.append(np.array(tmp_list2))
            
            
data=data[0:-2]            
            
#%% Data editing
df = pd.DataFrame()
br_merenja=[]
names=[]
iterations=[]
tmp_iterations=[]

# Separating ID and Name
for i in range(len(data)): 
    tmp=data[i]
    if(tmp[0]=='-' and tmp[1].isnumeric()):
           tmp=tmp.replace('-','')
           br_merenja.append(tmp) 
            
    elif(tmp[0]=='-' and tmp[1].isupper()):
           tmp=tmp.replace('-','')
           names.append(tmp)
           

          
# Separation of iterations from each measurement    

for i in range(2, len(data)):
    tmp=data[i]
    if(tmp[0]=='-' and tmp[1].isnumeric()):
        continue
    elif(tmp[0]=='-' and tmp[1].isupper()):
        # a=tmp_iterations[0]
        # for i in range(1, len(tmp_iterations)):
        #    a=np.concatenate((a, tmp_iterations[i])) 
        iterations.append(tmp_iterations)
        tmp_iterations=[]
        
    else: 
        tmp_iterations.append(data[i])   
        
iterations.append(tmp_iterations) # Manual addition to the last measurement
        

df['ID']=br_merenja
df['Person']=names
df['Samples']=iterations


#%% Characteristics
from statistics import median, mean, variance, stdev
from math import atan2
from scipy.stats import moment

speed_average=[]
speed_median=[]
speed_max =[]
speed_variance=[]
speed_tmp=[]
speed =[]
speed_stdev=[]
speed_moment=[]

# Speed
for i in range(len(df)):
    speed_tmp=[0] # Почетна Вредност
    for j in range(len(df['Samples'].iloc[i])-1):
      x1=df['Samples'].iloc[i][j]
      x2=df['Samples'].iloc[i][j+1]
      s=np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)
      t=x2[2]
      v=s/t
      speed_tmp.append(v)
      
      
    speed_avg= mean(speed_tmp)
    speed_med=median(speed_tmp)
    speed_mx=max(speed_tmp)
    speed_var=variance(speed_tmp)
    speed_st = stdev(speed_tmp)
    speed_mom=moment(speed_tmp)
    speed_average.append(speed_avg)
    speed_median.append(speed_med)
    speed_max.append(speed_mx)
    speed_variance.append(speed_var)
    speed.append(speed_tmp)
    speed_stdev.append(speed_st)
    speed_moment.append(speed_mom)
    
    

# Acceleration 
acc_temp=[]  
acc_average=[]
acc_median=[]
acc_max=[]
acc_variance=[]
acceleration=[]
acc_stdev=[]
acc_moment=[]

for i in range (len(speed)):
  
  
  acc_temp=[0] # Start value
  for j in range(0, len(df['Samples'].iloc[i])-1):
      x=df['Samples'].iloc[i][j+1]
      t=x[2]
      acc=np.abs(speed[i][j+1]-speed[i][j])/t
      acc_temp.append(acc)
      
  acc_avg=mean(acc_temp)
  acc_med=median(acc_temp)
  acc_mx=max(acc_temp)
  acc_var=variance(acc_temp)
  acc_st=stdev(acc_temp)
  acc_mom=moment(acc_temp)
  acc_average.append(acc_avg)
  acc_median.append(acc_med)
  acc_max.append(acc_mx)
  acceleration.append(acc_temp)
  acc_variance.append(acc_var)
  acc_stdev.append(acc_st)
  acc_moment.append(acc_mom)
  
# Jerk 
jerk_temp=[]
jerk_average=[]
jerk_median=[]
jerk_max=[]
jerk_variance=[]
jerk_stdev=[]
jerk_moment=[]

for i in range (len(acceleration)):
  
  jerk_temp=[0] # Start value
  for j in range(0, len(df['Samples'].iloc[i])-1):
      x=df['Samples'].iloc[i][j+1]
      t=x[2]
      jerk=np.abs(acceleration[i][j+1]-acceleration[i][j])/t
      jerk_temp.append(jerk)  
  
  jerk_avg=mean(jerk_temp)
  jerk_med=median(jerk_temp)
  jerk_mx=max(jerk_temp)
  jerk_var=variance(jerk_temp)
  jerk_st=stdev(jerk_temp)
  jerk_mom=moment(jerk_temp)
  jerk_average.append(jerk_avg)
  jerk_median.append(jerk_med)
  jerk_max.append(jerk_mx)
  jerk_variance.append(jerk_var)
  jerk_stdev.append(jerk_st)
  jerk_moment.append(jerk_mom)
    
# Distance 
distance_tmp=[]
distance_average=[]
distance_median=[]
distance_max=[]
distance_variance=[]
distance_stdev=[]
distance_moment=[]
  
for i in range(len(df)):
    distance_tmp=[0] # Start value
    for j in range(len(df['Samples'].iloc[i])-1):
      x1=df['Samples'].iloc[i][j]
      x2=df['Samples'].iloc[i][j+1]
      s=np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)
      distance_tmp.append(s)
      
    distance_avg=mean(distance_tmp)
    distance_med=median(distance_tmp)
    distance_mx=max(distance_tmp)
    distance_var=variance(distance_tmp)
    distance_st=stdev(distance_tmp)
    distance_mom=moment(distance_tmp)
    distance_average.append(distance_avg)
    distance_median.append(distance_med)
    distance_max.append(distance_mx)
    distance_variance.append(distance_var)
    distance_stdev.append(distance_st)
    distance_moment.append(distance_mom)
    
    

# Session Lenght 
ses_lenght=[]    
for i in range(len(df)):
    ses_lenght.append(len(df['Samples'].iloc[i]))

# x,y
x_max=[]
x_average=[]
x_median=[]
x_variance=[]
x_stdev=[]
x_moment=[]
y_max=[]
y_average=[]
y_median=[]
y_variance=[]
y_stdev=[]
y_moment=[]
x_tmp=[]
y_tmp=[]
area_avg=[]
for i in range(len(df)):
    x_tmp=[]
    y_tmp=[]
    for j in range(len(df['Samples'].iloc[i])):
       z=df['Samples'].iloc[i][j]
       x=z[0]
       y=z[1]
       x_tmp.append(x)
       y_tmp.append(y)
    
    x_mx=max(x_tmp)
    
    x_avg=mean(x_tmp)
    x_med=median(x_tmp)
    x_st=stdev(x_tmp)
    x_var=variance(x_tmp)
    x_mom=moment(x_tmp)
    
    y_mx=max(y_tmp)
    
    area=x_mx*y_mx
    
    area_avg.append(area)
    y_avg=mean(y_tmp)
    y_med=median(y_tmp)
    y_st=stdev(y_tmp)
    y_var=variance(y_tmp)
    y_mom=moment(y_tmp)
    
    x_max.append(x_mx)
    
    x_average.append(x_avg)
    x_median.append(x_med)
    x_stdev.append(x_st)
    x_variance.append(x_var)
    x_moment.append(x_mom)
    
    y_max.append(y_mx)
    
    y_average.append(y_avg)
    y_median.append(y_med)
    y_stdev.append(y_st)
    y_variance.append(y_var)
    y_moment.append(y_mom)
    
# Time 
time_average=[]
time_median=[]
time_tmp=[]
time_sum=[]
time_stdev=[]
time_variance=[]
time_moment=[]

    
for i in range(len(df)):
    time_tmp=[] # Start Value
    for j in range(0, len(df['Samples'].iloc[i])):   
       x=df['Samples'].iloc[i][j]
       t=x[2]
       time_tmp.append(t)
    
    t_avg=mean(time_tmp)
    t_med=median(time_tmp)
    t_s=sum(time_tmp)
    t_st=np.std(time_tmp)
    t_var=np.var(time_tmp)
    t_mom=moment(time_tmp)
    
    time_average.append(t_avg)
    time_median.append(t_med)
    time_sum.append(t_s)
    time_variance.append(t_var)
    time_stdev.append(t_st)
    time_moment.append(t_mom)
    

    
#%% Defining the new data in one df   
    
df_new=pd.DataFrame()

df_new['Person']=df['Person']
df_new['Average Speed']=speed_average
df_new['Median Speed']=speed_median
df_new['Max Speed']=speed_max
df_new['Variance Speed']=speed_variance
df_new['Std Speed']=speed_stdev
df_new['Average Acceleration']=acc_average
df_new['Median Acceleration']=acc_median
df_new['Max Acceleration']=acc_max
df_new['Acceleration Variance']=acc_variance
df_new['Acceleration Std']=acc_stdev
df_new['Average Jerk']=jerk_average
df_new['Median Jerk']=jerk_median
df_new['Max Jerk']=jerk_max
df_new['Jerk Variance']=jerk_variance
df_new['Jerk Std']=jerk_stdev
df_new['Average Distance']=distance_average
df_new['Median Distance']=distance_median
df_new['Max Distance']=distance_max
df_new['Distance Variance']=distance_variance
df_new['Distance Std']=distance_stdev
df_new['Session Lenght']=ses_lenght
df_new['x_average']=x_average
df_new['x_median']=x_median
df_new['x_max']=x_max
df_new['x_variance']=x_variance
df_new['x_std']=x_stdev
df_new['y_average']=y_average
df_new['y_median']=y_median
df_new['y_max']=y_max
df_new['y_variance']=y_variance
df_new['y_std']=y_stdev
df_new['Average Time']=time_average
df_new['Median Time']=time_median
df_new['Sum Time']=time_sum
df_new['Variance Time']=time_variance
df_new['Std Time']=time_stdev
df_new['Max Area']=area_avg




#%% Corelation
import seaborn as sns 
cor=df_new.corr().abs()
plt.figure(figsize=(10,6))
# sns.heatmap(cor, annot=True)
upper_tri = cor.where(np.triu(np.ones(cor.shape),k=1).astype(np.bool))

sns.heatmap(upper_tri, annot=True)

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

df_new.drop(to_drop, axis=1, inplace=True)

#%% Power Transformer

from sklearn.preprocessing import PowerTransformer

# Label encoding
from sklearn.preprocessing import LabelEncoder

labelencoder_X=LabelEncoder()
labels=df_new['Person']
labels=labelencoder_X.fit_transform(labels) 

df_new['Person']=labels

X=df_new.drop(columns=['Person'])
y=df_new['Person']

pt= PowerTransformer(method='yeo-johnson')

X_pt=pt.fit_transform(X)
columns=[]
for col_name in df_new.columns: 
    columns.append(col_name)

X=pd.DataFrame(X_pt, columns=columns[1:])

#%% XG Boost

from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler 

#  Label encoding
from sklearn.preprocessing import LabelEncoder

labelencoder_X=LabelEncoder()
labels=df_new['Person']
labels=labelencoder_X.fit_transform(labels) 

df_new['Person']=labels

X=df_new.drop(columns=['Person'])
y=df_new['Person']


scaler=StandardScaler()

X1=scaler.fit_transform(X)


columns=[]
for col_name in df_new.columns: 
    columns.append(col_name)

X=pd.DataFrame(X1, columns=columns[1:])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
model=XGBClassifier()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
# predictions=[round(value) for value in y_pred]

accuracy=accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))   

  
#%% Tuning  parameters

model = XGBClassifier()
learning_rate=[0.01, 0.05, 0.1, 0.3]
n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8]
colsample_bytree=[0.5, 0.6, 0.7, 0.8]
subsample=[0.8, 0.9, 1]


print(max_depth)
param_grid = dict(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, colsample_bytree=colsample_bytree, subsample=subsample)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))


#%%
    
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#%% 
xgb1 = xgb.sklearn.XGBClassifier(learning_rate=0.05,
 colsample_bytree= 0.7, max_depth= 4, 
 n_estimators= 200, subsample= 0.8
 )

#%% 
# model=XGBClassifier()


xgb1.fit(X_train, y_train)

y_pred=xgb1.predict(X_test)
predictions=[round(value) for value in y_pred]

accuracy=accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print('Training set score: {:.4f}'.format(xgb1.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(xgb1.score(X_test, y_test))) 

#%% Information Gain

from sklearn.feature_selection import mutual_info_classif
X=df_new.drop(columns=['Person'])
importances=mutual_info_classif(X,y)    
feat_importances=pd.Series(importances, df_new.columns[0:len(df_new.columns)-1])
feat_importances.plot(kind='barh', color='teal')
plt.show()





#%% Backward Feature Selection 

from mlxtend.feature_selection import SequentialFeatureSelector 
ffs=SequentialFeatureSelector(xgb1, k_features='best', forward=False, n_jobs=-1)
ffs.fit(X,y)
features=list(ffs.k_feature_names_)
print(features)
#%%
X1=X[features]
# Normalization
from sklearn.preprocessing import StandardScaler 

scaler=StandardScaler()

X1=scaler.fit_transform(X1)

X1=pd.DataFrame(X1, columns=(
                            'Average Speed', 
                            'Median Speed', 
                            'Max Speed', 
                            'Variance Speed', 
                            'Average Acceleration', 
                            'Median Acceleration', 
                               # 'Max Acceleration',
                               'Variance Acceleration',
                            'Average Jerk', 
                            'Median Jerk', 
                            # 'Max Jerk', 
                            # 'Variance Jerk',
                            'Average Distance',
                           
                            'Median Distance', 
                            'Max Distance', 
                            'Variance Distance',
                            'Session Lenght', 
                            'x_average',
                            'x median',
                             'x_max', 
                              'y_average', 
                            'y_max', 
                            'y_median', 
                             # 'Average Time', 
                            'Median Time', 
                            # 'Max Area', 
                            # 'Sum Time'
                           ))



X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=1/3, random_state=0)
model=XGBClassifier()
xgb1.fit(X_train, y_train)

y_pred=xgb1.predict(X_test)
# predictions=[round(value) for value in y_pred]

accuracy=accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print('Training set score: {:.4f}'.format(xgb1.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(xgb1.score(X_test, y_test)))   

#%% Ideal score with k=N
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import std
from sklearn.model_selection import LeaveOneOut

scores = cross_val_score(xgb1, X, y, scoring='accuracy', cv=LeaveOneOut(), n_jobs=1 )
ideal =mean(scores)
print('Ideal: %.3f' % ideal)

#%% Sensitivity analysis for k

folds = range (2, 31)
means = [] 
for k in folds: 
    cv = KFold ( n_splits=k, shuffle=True, random_state = 1)
    scores = cross_val_score(xgb1, X, y, scoring='accuracy', cv=cv, n_jobs=1 )
    k_mean=mean(scores)
    means.append(k_mean)

# plot 
plt.errorbar(folds, means, fmt='o')
plt.plot(folds, [ideal for _ in range(len(folds))], color='r')    
plt.show()


#%% Cross Validation 


cv=KFold(n_splits=15, random_state=1, shuffle= True )
scores = cross_val_score(xgb1, X, y, scoring='accuracy', cv=cv, n_jobs=1 )

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))   


#%% Test_Train 

X_train_all=[]
X_test_all=[]
y_train_all=[]
y_test_all=[]

for train_index, test_index in cv.split(X):
      
      X_train_all.append(X.iloc[train_index])
      X_test_all.append(X.iloc[test_index])
      y_train_all.append(y[train_index])
      y_test_all.append(y[test_index])
X_train=X_train_all[0]
X_test=X_test_all[0]
y_train=y_train_all[0]
y_test=y_test_all[0] 

#%% Training of the model

xgb1.fit(X_train, y_train)

y_pred=xgb1.predict(X_test)
# predictions=[round(value) for value in y_pred]

accuracy=accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print('Training set score: {:.4f}'.format(xgb1.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(xgb1.score(X_test, y_test)))   

#%% 

results = xgb1.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()
# plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.show()

#%% Loading unseen data


data = []

with open('mouse_test.txt', "r") as f:
    for line in f:
        if line[0] == '-':
            data.append(line)
        else:
            linija = line[1:-2] # Remove parenthisis
            tmp_list = linija.split(',')    # Remove commas
            tmp_list2 = [int(x) for x in tmp_list]  # string to int
            data.append(np.array(tmp_list2))
            
            
data=data[0:-2] 

#%% Editing unseen data

df = pd.DataFrame()
br_merenja=[]
names=[]
iterations=[]
tmp_iterations=[]

# Separation of ID and Name 
for i in range(len(data)): 
    tmp=data[i]
    if(tmp[0]=='-' and tmp[1].isnumeric()):
           tmp=tmp.replace('-','')
           br_merenja.append(tmp) 
            
    elif(tmp[0]=='-' and tmp[1]=='?'):
           tmp=tmp.replace('-','')
           names.append(tmp)
           

          
# Separation of iterations from each measurement    

for i in range(2, len(data)):
    tmp=data[i]
    if(tmp[0]=='-' and tmp[1].isnumeric()):
        continue
    elif(tmp[0]=='-' and tmp[1]=='?'):
        # a=tmp_iterations[0]
        # for i in range(1, len(tmp_iterations)):
        #    a=np.concatenate((a, tmp_iterations[i])) 
        iterations.append(tmp_iterations)
        tmp_iterations=[]
        
    else: 
        tmp_iterations.append(data[i])   
        
iterations.append(tmp_iterations) 
        

df['ID']=br_merenja
df['Person']=names
df['Samples']=iterations

#%% Characteristics of the non available data
from statistics import median, mean, variance, stdev
from math import atan2
from scipy.stats import moment

speed_average=[]
speed_median=[]
speed_max =[]
speed_variance=[]
speed_tmp=[]
speed =[]
speed_stdev=[]
speed_moment=[]

# Speed
for i in range(len(df)):
    speed_tmp=[0] # Start val
    for j in range(len(df['Samples'].iloc[i])-1):
      x1=df['Samples'].iloc[i][j]
      x2=df['Samples'].iloc[i][j+1]
      s=np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)
      t=x2[2]
      v=s/t
      speed_tmp.append(v)
      
      
    speed_avg= mean(speed_tmp)
    speed_med=median(speed_tmp)
    speed_mx=max(speed_tmp)
    speed_var=variance(speed_tmp)
    speed_st = stdev(speed_tmp)
    speed_mom=moment(speed_tmp)
    speed_average.append(speed_avg)
    speed_median.append(speed_med)
    speed_max.append(speed_mx)
    speed_variance.append(speed_var)
    speed.append(speed_tmp)
    speed_stdev.append(speed_st)
    speed_moment.append(speed_mom)
    
    

# Acceleration 
acc_temp=[]  
acc_average=[]
acc_median=[]
acc_max=[]
acc_variance=[]
acceleration=[]
acc_stdev=[]
acc_moment=[]

for i in range (len(speed)):
  
  
  acc_temp=[0] # Start value
  for j in range(0, len(df['Samples'].iloc[i])-1):
      x=df['Samples'].iloc[i][j+1]
      t=x[2]
      acc=np.abs(speed[i][j+1]-speed[i][j])/t
      acc_temp.append(acc)
      
  acc_avg=mean(acc_temp)
  acc_med=median(acc_temp)
  acc_mx=max(acc_temp)
  acc_var=variance(acc_temp)
  acc_st=stdev(acc_temp)
  acc_mom=moment(acc_temp)
  acc_average.append(acc_avg)
  acc_median.append(acc_med)
  acc_max.append(acc_mx)
  acceleration.append(acc_temp)
  acc_variance.append(acc_var)
  acc_stdev.append(acc_st)
  acc_moment.append(acc_mom)
  
# Jerk 
jerk_temp=[]
jerk_average=[]
jerk_median=[]
jerk_max=[]
jerk_variance=[]
jerk_stdev=[]
jerk_moment=[]

for i in range (len(acceleration)):
  
  jerk_temp=[0] # Start value
  for j in range(0, len(df['Samples'].iloc[i])-1):
      x=df['Samples'].iloc[i][j+1]
      t=x[2]
      jerk=np.abs(acceleration[i][j+1]-acceleration[i][j])/t
      jerk_temp.append(jerk)  
  
  jerk_avg=mean(jerk_temp)
  jerk_med=median(jerk_temp)
  jerk_mx=max(jerk_temp)
  jerk_var=variance(jerk_temp)
  jerk_st=stdev(jerk_temp)
  jerk_mom=moment(jerk_temp)
  jerk_average.append(jerk_avg)
  jerk_median.append(jerk_med)
  jerk_max.append(jerk_mx)
  jerk_variance.append(jerk_var)
  jerk_stdev.append(jerk_st)
  jerk_moment.append(jerk_mom)
    
# Distance 
distance_tmp=[]
distance_average=[]
distance_median=[]
distance_max=[]
distance_variance=[]
distance_stdev=[]
distance_moment=[]
  
for i in range(len(df)):
    distance_tmp=[0] # Start Value
    for j in range(len(df['Samples'].iloc[i])-1):
      x1=df['Samples'].iloc[i][j]
      x2=df['Samples'].iloc[i][j+1]
      s=np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)
      distance_tmp.append(s)
      
    distance_avg=mean(distance_tmp)
    distance_med=median(distance_tmp)
    distance_mx=max(distance_tmp)
    distance_var=variance(distance_tmp)
    distance_st=stdev(distance_tmp)
    distance_mom=moment(distance_tmp)
    distance_average.append(distance_avg)
    distance_median.append(distance_med)
    distance_max.append(distance_mx)
    distance_variance.append(distance_var)
    distance_stdev.append(distance_st)
    distance_moment.append(distance_mom)
    
    

# Session Lenght 
ses_lenght=[]    
for i in range(len(df)):
    ses_lenght.append(len(df['Samples'].iloc[i]))

# x,y
x_max=[]
x_average=[]
x_median=[]
x_variance=[]
x_stdev=[]
x_moment=[]
y_max=[]
y_average=[]
y_median=[]
y_variance=[]
y_stdev=[]
y_moment=[]
x_tmp=[]
y_tmp=[]
area_avg=[]
for i in range(len(df)):
    x_tmp=[]
    y_tmp=[]
    for j in range(len(df['Samples'].iloc[i])):
       z=df['Samples'].iloc[i][j]
       x=z[0]
       y=z[1]
       x_tmp.append(x)
       y_tmp.append(y)
    
    x_mx=max(x_tmp)
    
    x_avg=mean(x_tmp)
    x_med=median(x_tmp)
    x_st=stdev(x_tmp)
    x_var=variance(x_tmp)
    x_mom=moment(x_tmp)
    
    y_mx=max(y_tmp)
    
    area=x_mx*y_mx
    
    area_avg.append(area)
    y_avg=mean(y_tmp)
    y_med=median(y_tmp)
    y_st=stdev(y_tmp)
    y_var=variance(y_tmp)
    y_mom=moment(y_tmp)
    
    x_max.append(x_mx)
    
    x_average.append(x_avg)
    x_median.append(x_med)
    x_stdev.append(x_st)
    x_variance.append(x_var)
    x_moment.append(x_mom)
    
    y_max.append(y_mx)
    
    y_average.append(y_avg)
    y_median.append(y_med)
    y_stdev.append(y_st)
    y_variance.append(y_var)
    y_moment.append(y_mom)
    
# Time 
time_average=[]
time_median=[]
time_tmp=[]
time_sum=[]
time_stdev=[]
time_variance=[]
time_moment=[]

    
for i in range(len(df)):
    time_tmp=[] # Почетна Вредност
    for j in range(0, len(df['Samples'].iloc[i])):   
       x=df['Samples'].iloc[i][j]
       t=x[2]
       time_tmp.append(t)
    
    t_avg=mean(time_tmp)
    t_med=median(time_tmp)
    t_s=sum(time_tmp)
    t_st=np.std(time_tmp)
    t_var=np.var(time_tmp)
    t_mom=moment(time_tmp)
    
    time_average.append(t_avg)
    time_median.append(t_med)
    time_sum.append(t_s)
    time_variance.append(t_var)
    time_stdev.append(t_st)
    time_moment.append(t_mom)

#%% Defining a universal data frame 



df_new=pd.DataFrame()

df_new['Person']=df['Person']
df_new['Average Speed']=speed_average
df_new['Median Speed']=speed_median
df_new['Max Speed']=speed_max
df_new['Variance Speed']=speed_variance
df_new['Std Speed']=speed_stdev
df_new['Average Acceleration']=acc_average
df_new['Median Acceleration']=acc_median
df_new['Max Acceleration']=acc_max
df_new['Acceleration Variance']=acc_variance
df_new['Acceleration Std']=acc_stdev
df_new['Average Jerk']=jerk_average
df_new['Median Jerk']=jerk_median
df_new['Max Jerk']=jerk_max
df_new['Jerk Variance']=jerk_variance
df_new['Jerk Std']=jerk_stdev
df_new['Average Distance']=distance_average
df_new['Median Distance']=distance_median
df_new['Max Distance']=distance_max
df_new['Distance Variance']=distance_variance
df_new['Distance Std']=distance_stdev
df_new['Session Lenght']=ses_lenght
df_new['x_average']=x_average
df_new['x_median']=x_median
df_new['x_max']=x_max
df_new['x_variance']=x_variance
df_new['x_std']=x_stdev
df_new['y_average']=y_average
df_new['y_median']=y_median
df_new['y_max']=y_max
df_new['y_variance']=y_variance
df_new['y_std']=y_stdev
df_new['Average Time']=time_average
df_new['Median Time']=time_median
df_new['Sum Time']=time_sum
df_new['Variance Time']=time_variance
# df_new['Std Time']=time_stdev
df_new['Max Area']=area_avg

#%% Corelation
import seaborn as sns 
cor=df_new.corr().abs()
plt.figure(figsize=(10,6))
# sns.heatmap(cor, annot=True)
upper_tri = cor.where(np.triu(np.ones(cor.shape),k=1).astype(np.bool))

sns.heatmap(upper_tri, annot=True)

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

df_new.drop(to_drop, axis=1, inplace=True)


#%% Power Transformer

from sklearn.preprocessing import PowerTransformer

# Label encoding
from sklearn.preprocessing import LabelEncoder

labelencoder_X=LabelEncoder()
labels=df_new['Person']
labels=labelencoder_X.fit_transform(labels) 

df_new['Person']=labels

X=df_new.drop(columns=['Person'])
y=df_new['Person']



X_pt=pt.transform(X)
columns=[]
for col_name in df_new.columns: 
    columns.append(col_name)

X_test=pd.DataFrame(X_pt, columns=columns[1:])

#%% Testing of the algorythm

X=df_new.drop(columns=['Person'])

X_pt=scaler.transform(X)
columns=[]
for col_name in df_new.columns: 
    columns.append(col_name)

X_test=pd.DataFrame(X_pt, columns=columns[1:])
                          
y_pred=xgb1.predict(X_test)

# predictions=[round(value) for value in y_pred]

#%% Output format 

index = []
output = []

for i in range(0, len(y_pred)):
    index.append(i+1)
    if y_pred[i] == 0:
        output.append('Dushko')
    elif y_pred[i] == 1:
        output.append('Filip')
    elif y_pred[i] == 2:
        output.append('Stefan')
    elif y_pred[i] == 3:
        output.append('Vesna')
        
out = pd.DataFrame(index, columns=['id'])
out['Person'] = output

out.to_csv('outputs-upm.csv')