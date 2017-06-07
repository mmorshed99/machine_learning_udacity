# Import libraries necessary for this project

import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
from sklearn import cross_validation
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn import grid_search
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import PolynomialFeatures
from bokeh.plotting import figure,output_file,show

def performance_metric(y_true, y_predict):

   score = r2_score(y_true, y_predict)
   return score

def linear_regression_model(X, y):

   regr = linear_model.LinearRegression()
   return regr.fit(X, y)

def fit_sgd(X, y):

   regr = SGDRegressor(n_iter = 5000,alpha = 0.00001)
   return regr.fit(X, y)

def fit_decision_tree_model(X, y):
   cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 1)
   regr = DecisionTreeRegressor(random_state=1)
   params = {'max_depth':(8,9,10,11,12,13,14,15)}
   scoring_fnc = make_scorer(performance_metric, greater_is_better=True)
   grid = grid_search.GridSearchCV(regr,params,scoring=scoring_fnc,cv=cv_sets)
   grid = grid.fit(X, y)
   return grid.best_estimator_

def fit_randomforest_initial(X, y):
   regr = RandomForestRegressor()
   return regr.fit(X, y)


def fit_randomforest(X, y):
   #regr = RandomForestRegressor(n_estimators = 25, max_features=100,max_depth=12)
   #regr = RandomForestRegressor(n_estimators = 50, max_features=125,max_depth=16)  
   #regr = RandomForestRegressor(n_estimators = 75, max_features=150,max_depth=20) 
   #regr = RandomForestRegressor(n_estimators = 100, max_features=175,max_depth=22) 
   #regr = RandomForestRegressor(n_estimators = 125, max_features=200,max_depth=24)
   #regr = RandomForestRegressor(n_estimators = 125, max_features=225,max_depth=22) 
   #regr = RandomForestRegressor(n_estimators = 125, max_features=250,max_depth=22) 
   #regr = RandomForestRegressor(n_estimators = 125, max_features=275,max_depth=22) 
   #regr = RandomForestRegressor(n_estimators = 125, max_features=300,max_depth=22) 
   #regr = RandomForestRegressor(n_estimators = 125, max_features=325,max_depth=22)
   #regr = RandomForestRegressor(n_estimators = 125, max_features=335,max_depth=16)  
   regr = RandomForestRegressor(n_estimators = 125, max_features=335,max_depth=22)
   return regr.fit(X, y)

def fit_neighbors(X, y,n):
   regr = KNeighborsRegressor(n_neighbors=n)
   return regr.fit(X, y)

def bayes_model(X, y):

   regr = linear_model.BayesianRidge()
   return regr.fit(X, y)
data = pd.read_csv('allstate_claim.csv')
#data = pd.read_csv('medium.csv')

####Removing entries with claim value more than $30,000 ###
print "Allstate claims dataset has {} data points with {} variables each including id and claim value attributes.".format(*data.shape)

data = data[data.loss < 30000.0]

loss = data['loss']
features = data.drop('loss', axis = 1)
le = preprocessing.LabelEncoder()

####### converting the features described by letters into encoded values using LabelEncoder() function #######
####### New data frame will be called modified_data ######

dict = {}

dict['temp1'] = features['cat1']  

modified_data = pd.DataFrame(dict['temp1'],index=None)

#### Label Encoder was chosen over onehotencoder to limit the number of features...it would increase to 1000 from 130 ###

for i in range(2,117):
   dict['temp'] = "cat"+str(i)
   dict['temp'] = getattr(features,dict['temp'])
   temp_frame =pd.DataFrame(dict['temp'],index=None)
   modified_data = pd.concat([modified_data, temp_frame], axis =1, join = 'inner')

modified_data = modified_data.apply(LabelEncoder().fit_transform)

for i in range(1,15):
   dict['temp'] = "cont"+str(i)
   dict['temp'] = getattr(features,dict['temp'])
   temp_frame = pd.DataFrame(dict['temp'],index=None)
   modified_data = pd.concat([modified_data, temp_frame], axis =1, join = 'inner')


##### Getting statistics of the dataset ######

print "Allstate claims dataset has {} data points with {} variables each after removing higher claim values.".format(*modified_data.shape) 

maximum_loss = np.amax(loss)
print "Maximum loss: ${:,.2f}".format(maximum_loss) 

minimum_loss = np.amin(loss)
print "Minimum loss: ${:,.2f}".format(minimum_loss)

mean_loss = np.mean(loss)
print "Mean loss: ${:,.2f}".format(mean_loss)

median_loss = np.median(loss)
print "Median loss: ${:,.2f}".format(median_loss)

std_dev_loss = np.std(loss)
print "Standard deviation of loss: ${:,.2f}".format(std_dev_loss)

#print modified_data.axes

print "cont1 minimum value"
cont1_min = np.amin(data['cont1'])

print cont1_min

print "cont1 maximum value"
cont1_max = np.amax(data['cont1'])

print cont1_max

#print modified_data.axes

#### Using 1% of the dataset for feature selection, 10% was also tried, but it didn't give performance improvement ####
#X_train, X_init, y_train, y_init = cross_validation.train_test_split(modified_data,loss,test_size=0.1,random_state=1)
X_train, X_init, y_train, y_init = cross_validation.train_test_split(modified_data,loss,test_size=0.01,random_state=1)
#print X_train.axes

#### Using Linear regression for feature selection and selecting best 'K' features ####

my_filter = SelectKBest(f_regression)
clf = linear_model.LinearRegression()

my_tree = Pipeline([('myfilter',my_filter), ('tree', clf)])
 
my_tree.set_params(myfilter__k=125).fit(X_init,y_init)

##### features_weight is True or False based on the selection. If False, that feature is removed from the dataset

features_weight = my_tree.named_steps['myfilter'].get_support()

#print features_weight

#print len(features_weight)
last_index_of_feat_weight_used = -1

for i in range(1,117): 
   if features_weight[i] == False:
      dict['temp'] = "cat"+str(i)
      X_train = X_train.drop(dict['temp'], axis = 1)
      last_index_of_feat_weight_used = i-1   
      
j = 1
for i in range(last_index_of_feat_weight_used+1,len(features_weight)):
   if features_weight[i] == False:
      dict['temp'] = "cont"+str(j)
      X_train =X_train.drop(dict['temp'], axis = 1)
   j += 1


#### Converting the dataset to polynomal features with order of 2 ####


poly = PolynomialFeatures(degree=2)                                   
X_train = poly.fit_transform(X_train)       
X_train = pd.DataFrame(X_train,index=None)     

#### Separating 20% of the remaining data for using as Test Set ####

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_train,y_train,test_size=0.2,random_state=1)

#regressor = SVR(C=10, epsilon=0.1,kernel='linear')
#regressor = regressor.fit(X_train,y_train)

#regressor = linear_regression_model(X_train, y_train)
#y_pred = regressor.predict(X_test)

#print performance_metric(y_test, y_pred)

#regressor = fit_decision_tree_model(X_train, y_train)
#y_pred = regressor.predict(X_test)

#regressor = fit_sgd(X_train, y_train)                                                                                                                             
#y_pred = regressor.predict(X_test)             


#regressor = fit_neighbors(X_train, y_train,5)
#y_pred = regressor.predict(X_test)
#for n in [10,15,20,25,30]:]
#regressor = fit_neighbors(X_train, y_train,n)                                                                                                                                    
#y_pred_temp = regressor.predict(X_test)
#if performance_metric(y_test,y_pred_temp) > performance_metric(y_test,y_pred):
###y_pred=y_pred_temp

#### Calling random forest function defined at the top to create the model with training set ####

regressor = fit_randomforest(X_train, y_train)                                                                                                                                                                    
y_pred = regressor.predict(X_test)        

#regressor = bayes_model(X_train,y_train)
#y_pred = regressor.predict(X_test)

print "training set r^2 value:"
y_train_pred = regressor.predict(X_train)
print performance_metric(y_train, y_train_pred)

print "test set r^2 value:"

print performance_metric(y_test, y_pred)    

#### plotting the actual and model output comparison ####

output_file("prediction_vs_real_output.html")
p = figure(plot_width=1000,plot_height=1000,title='prediction vs real output',x_axis_label = 'Actual Claim',y_axis_label = 'Model Predicted Claim')


y_pred=np.append(y_pred, [30000], axis=None)
y_test=np.append(y_test, [30000], axis=None)

p.circle(y_test,y_pred)
show(p)

#### Selecting arbitraty data points from test set and getting the model output to create comparison table #####

X_test = pd.DataFrame(X_test,index=None)
X_sample_test = X_test.iloc[[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]]

y_test = pd.DataFrame(y_test,index=None)
y_sample_test =y_test.iloc[[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]]

print "print actual and model outputs for few cases"

print "real sample outputs:"
print y_sample_test

print "sample outputs from model:"
print regressor.predict(X_sample_test)
