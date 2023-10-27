# python-predicted-value
Logistic regression, Support vector machine, decision tree, knn
import pandas as pd                                    # needed to read the data                            
import numpy as np                                     # needed for arrays                          
from sklearn.model_selection import train_test_split   # splits database                            
from sklearn.preprocessing import StandardScaler       # standardize data                           
from sklearn.linear_model import Perceptron            # the Perceptron algorithm                           
from sklearn.ensemble import RandomForestClassifier    # the Random Forest algorithm                            
from sklearn.svm import SVC                            # the SVM algorithm                          
from sklearn.linear_model import LogisticRegression    # the Logistic Regression algorithm                          
from sklearn.neighbors import KNeighborsClassifier     # the knn algorithm                          
from sklearn.tree import DecisionTreeClassifier        # the Decision Tree algorithm                            
from sklearn.metrics import accuracy_score             # grade the results                          
from sklearn.metrics import confusion_matrix
#import scikitplot as skplt
​
heart = pd.read_csv('CLABSINEW1data.csv',encoding= 'unicode_escape')                     # load the data set                            
y = heart.iloc[:,3].values  # extract the classifications
​
​
# perceptron linear                         
                            
X_ppn = heart.iloc[:,[4, 12]].values          # separate the features we want                           
                            
# split the problem into train and test                         
# this will yield 70% training and 30% test                         
# random_state allows the split to be reproduced                            
# stratify=y not used in this case                          
X_train_ppn, X_test_ppn, y_train_ppn, y_test_ppn = train_test_split(X_ppn,y,test_size=0.3,random_state=0)                           
​
# scale X by removing the mean and setting the variance to 1 on all features.                           
# the formula is z=(x-u)/s where u is the mean and s is the standard deviation.                         
# (mean and standard deviation may be overridden with options...)                           
sc_ppn = StandardScaler()                          # create the standard scalar                         
sc_ppn.fit(X_train_ppn)                            # compute the required transformation                            
X_train_std_ppn = sc_ppn.transform(X_train_ppn)    # apply to the training data                         
X_test_std_ppn = sc_ppn.transform(X_test_ppn)      # and SAME transformation of test data!!!                            
​
# epoch is one forward and backward pass of all training samples (also an iteration)                            
# eta0 is rate of convergence                           
# max_iter, tol, if it is too low it is never achieved                          
# and continues to iterate to max_iter when above tol                           
# fit_intercept, fit the intercept or assume it is 0                            
# slowing it down is very effective, eta is the learning rate                           
ppn = Perceptron(max_iter=10, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=True)                           
ppn.fit(X_train_std_ppn, y_train_ppn)                               
​
-- Epoch 1
Norm: 0.00, NNZs: 1, Bias: -0.001000, T: 41, Avg. loss: 0.000000
Total training time: 0.00 seconds.
-- Epoch 2
Norm: 0.00, NNZs: 1, Bias: -0.001000, T: 82, Avg. loss: 0.000000
Total training time: 0.00 seconds.
-- Epoch 3
Norm: 0.00, NNZs: 1, Bias: -0.001000, T: 123, Avg. loss: 0.000000
Total training time: 0.00 seconds.
-- Epoch 4
Norm: 0.00, NNZs: 1, Bias: -0.001000, T: 164, Avg. loss: 0.000000
Total training time: 0.00 seconds.
-- Epoch 5
Norm: 0.00, NNZs: 1, Bias: -0.001000, T: 205, Avg. loss: 0.000000
Total training time: 0.00 seconds.
-- Epoch 6
Norm: 0.00, NNZs: 1, Bias: -0.001000, T: 246, Avg. loss: 0.000000
Total training time: 0.00 seconds.
Convergence after 6 epochs took 0.00 seconds
Perceptron(eta0=0.001, max_iter=10, verbose=True)
y_pred_ppn = ppn.predict(X_test_std_ppn)    # now try with the test data                        
                        
# combine the train and test data                       
# vstack puts first array above the second in a vertical stack                      
# hstack puts first array to left of the second in a horizontal stack                       
X_combined_std_ppn = np.vstack((X_train_std_ppn, X_test_std_ppn))                       
y_combined_ppn = np.hstack((y_train_ppn, y_test_ppn))                           
​
# check results on combined data                    
y_combined_pred_ppn = ppn.predict(X_combined_std_ppn)                   
                    
print('\nPerceptron :')                 
​

Perceptron :
# Note that this only counts the samples where the predicted value was wrong                            
print('Misclassified samples: %d' % (y_test_ppn != y_pred_ppn).sum())                           
                            
print('Accuracy: %.2f' % accuracy_score(y_test_ppn, y_pred_ppn))                            
print('Misclassified combined samples: %d' % (y_combined_ppn != y_combined_pred_ppn).sum())                         
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_ppn, y_combined_pred_ppn))                          
print('')                           
​
Misclassified samples: 0
Accuracy: 1.00
Misclassified combined samples: 0
Combined Accuracy: 1.00

cm = confusion_matrix(y_test_ppn, y_pred_ppn)                       
precision = cm[0][0]/(cm[0][0]+cm[0][1])*100                        
print('Positive Predicted value:',str(precision))                       
#skplt.metrics.plot_confusion_matrix(y_test_ppn, y_pred_ppn,figsize=(8,8))                      
print('\n')                     
                        
​
Positive Predicted value: 100.0


# logistic regression                           
X_lr = heart.iloc[:,[4, 12]].values           # separate the features we want                           
                            
# split the problem into train and test                         
# this will yield 70% training and 30% test                         
# random_state allows the split to be reproduced                            
# stratify=y not used in this case                          
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr,y,test_size=0.3,random_state=0)                            
​
# scale X by removing the mean and setting the variance to 1 on all features.                           
# the formula is z=(x-u)/s where u is the mean and s is the standard deviation.                         
# (mean and standard deviation may be overridden with options...)                           
sc_lr = StandardScaler()                         # create the standard scalar                           
sc_lr.fit(X_train_lr)                            # compute the required transformation                          
X_train_std_lr = sc_lr.transform(X_train_lr)     # apply to the training data                           
X_test_std_lr = sc_lr.transform(X_test_lr)       # and SAME transformation of test data!!!                          
                            
​
# create logistic regression component.                         
# C is the inverse of the regularization strength. Smaller -> stronger!                         
# C is used to penalize extreme parameter weights.                          
# solver is the particular algorithm to use                         
# multi_class determines how loss is computed - ovr -> binary problem for each label                            
lr = LogisticRegression(C=10, solver='liblinear', multi_class='ovr', random_state=0)                            
lr.fit(X_train_std_lr, y_train_lr)                          
​
LogisticRegression(C=10, multi_class='ovr', random_state=0, solver='liblinear')
y_pred_lr = lr.predict(X_test_std_lr)  # now try with the test data                     
                        
# combine the train and test data                       
# vstack puts first array above the second in a vertical stack                      
# hstack puts first array to left of the second in a horizontal stack                       
X_combined_std_lr = np.vstack((X_train_std_lr, X_test_std_lr))                      
y_combined_lr = np.hstack((y_train_lr, y_test_lr))                      
                        
# check results on combined data                        
y_combined_pred_lr = lr.predict(X_combined_std_lr)                      
                        
print('Logistic Regression :')                      
​
Logistic Regression :
# Note that this only counts the samples where the predicted value was wrong                            
print('Misclassified samples: %d' % (y_test_lr != y_pred_lr).sum())                         
                            
print('Accuracy: %.2f' % accuracy_score(y_test_lr, y_pred_lr))                          
print('Misclassified combined samples: %d' % (y_combined_lr != y_combined_pred_lr).sum())                           
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_lr, y_combined_pred_lr))                            
print('')                           
                            
cm = confusion_matrix(y_test_lr, y_pred_lr)                         
precision = cm[0][0]/(cm[0][0]+cm[0][1])*100                            
print('Positive Predicted value:',str(precision))                           
#skplt.metrics.plot_confusion_matrix(y_test_lr, y_pred_lr,figsize=(8,8))                            
print('\n')                         
​
Misclassified samples: 0
Accuracy: 1.00
Misclassified combined samples: 0
Combined Accuracy: 1.00

Positive Predicted value: 100.0


# SVM                           
X_svm = heart.iloc[:,[4, 12]].values           # separate the features we want                          
                            
# split the problem into train and test                         
# this will yield 70% training and 30% test                         
# random_state allows the split to be reproduced                            
# stratify=y not used in this case                          
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm,y,test_size=0.3,random_state=0)                           
​
# scale X by removing the mean and setting the variance to 1 on all features.                           
# the formula is z=(x-u)/s where u is the mean and s is the standard deviation.                         
# (mean and standard deviation may be overridden with options...)                           
sc_svm = StandardScaler()                          # create the standard scalar                         
sc_svm.fit(X_train_svm)                            # compute the required transformation                            
X_train_std_svm = sc_svm.transform(X_train_svm)    # apply to the training data                         
X_test_std_svm = sc_svm.transform(X_test_svm)      # and SAME transformation of test data!!!                            
​
# kernal - specify the kernal type to use                           
# C - the penalty parameter - it controls the desired margin size                           
svm = SVC(kernel='linear', C=1.0, random_state=0)                           
svm.fit(X_train_std_svm, y_train_svm)                           
                            
y_pred_svm = svm.predict(X_test_std_svm)  # now try with the test data                          
                            
# combine the train and test data                           
# vstack puts first array above the second in a vertical stack                          
# hstack puts first array to left of the second in a horizontal stack                           
X_combined_std_svm = np.vstack((X_train_std_svm, X_test_std_svm))                           
y_combined_svm = np.hstack((y_train_svm, y_test_svm))                           
                            
# check results on combined data                            
y_combined_pred_svm = svm.predict(X_combined_std_svm)                           
                            
print('Support Vector Machine :')                           
​
Support Vector Machine :
# Note that this only counts the samples where the predicted value was wrong                            
print('Misclassified samples: %d' % (y_test_svm != y_pred_svm).sum())                           
                            
print('Accuracy: %.2f' % accuracy_score(y_test_svm, y_pred_svm))                            
print('Misclassified combined samples: %d' % (y_combined_svm != y_combined_pred_svm).sum())                         
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_svm, y_combined_pred_svm))                          
print('')                           
                            
cm = confusion_matrix(y_test_svm, y_pred_svm)                           
precision = cm[0][0]/(cm[0][0]+cm[0][1])*100                            
print('Positive Predicted value:',str(precision))                           
#skplt.metrics.plot_confusion_matrix(y_test_svm, y_pred_svm,figsize=(8,8))                          
print('\n')                         
​
Misclassified samples: 0
Accuracy: 1.00
Misclassified combined samples: 0
Combined Accuracy: 1.00

Positive Predicted value: 100.0


# decision tree                         
X_tree = heart.iloc[:,[4, 12]].values           # separate the features we want                         
                            
# split the problem into train and test                         
# this will yield 70% training and 30% test                         
# random_state allows the split to be reproduced                            
# stratify=y not used in this case                          
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree,y,test_size=0.3,random_state=0)                          
                            
# scale X by removing the mean and setting the variance to 1 on all features.                           
# the formula is z=(x-u)/s where u is the mean and s is the standard deviation.                         
# (mean and standard deviation may be overridden with options...)                           
sc_tree = StandardScaler()                           # create the standard scalar                           
sc_tree.fit(X_train_tree)                            # compute the required transformation                          
X_train_std_tree = sc_tree.transform(X_train_tree)   # apply to the training data                           
X_test_std_tree = sc_tree.transform(X_test_tree)     # and SAME transformation of test data!!!                          
                            
# create the classifier and train it                            
tree = DecisionTreeClassifier(criterion='entropy',max_depth=5 ,random_state=0)                          
tree.fit(X_train_tree,y_train_tree)                         
                            
y_pred_tree = tree.predict(X_test_std_tree)  # now try with the test data                           
                            
# combine the train and test data                           
# vstack puts first array above the second in a vertical stack                          
# hstack puts first array to left of the second in a horizontal stack                           
X_combined_tree = np.vstack((X_train_tree, X_test_tree))                            
y_combined_tree = np.hstack((y_train_tree, y_test_tree))                            
                            
# check results on combined data                            
y_combined_pred_tree = tree.predict(X_combined_tree)                            
                            
print('Decision Tree :')                            
​
Decision Tree :
# Note that this only counts the samples where the predicted value was wrong                            
print('Misclassified samples: %d' % (y_test_tree != y_pred_tree).sum())                         
                            
print('Accuracy: %.2f' % accuracy_score(y_test_tree, y_pred_tree))                          
print('Misclassified combined samples: %d' % (y_combined_tree != y_combined_pred_tree).sum())                           
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_tree, y_combined_pred_tree))                            
print('')                           
                            
cm = confusion_matrix(y_test_tree, y_pred_tree)                         
precision = cm[0][0]/(cm[0][0]+cm[0][1])*100                            
print('Positive Predicted value:',str(precision))                           
#skplt.metrics.plot_confusion_matrix(y_test_tree, y_pred_tree,figsize=(8,8))                            
print('\n')                         
​
Misclassified samples: 0
Accuracy: 1.00
Misclassified combined samples: 0
Combined Accuracy: 1.00

Positive Predicted value: 100.0


# knn                           
X_knn = heart.iloc[:,[4, 12]].values           # separate the features we want                          
                            
# split the problem into train and test                         
# this will yield 70% training and 30% test                         
# random_state allows the split to be reproduced                            
# stratify=y not used in this case                          
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn,y,test_size=0.3,random_state=0)                           
                            
# scale X by removing the mean and setting the variance to 1 on all features.                           
# the formula is z=(x-u)/s where u is the mean and s is the standard deviation.                         
# (mean and standard deviation may be overridden with options...)                           
sc_knn = StandardScaler()                          # create the standard scalar                         
sc_knn.fit(X_train_knn)                            # compute the required transformation                            
X_train_std_knn = sc_knn.transform(X_train_knn)    # apply to the training data                         
X_test_std_knn = sc_knn.transform(X_test_knn)      # and SAME transformation of test data!!!                            
                            
# create the classifier and fit it                          
# using 10 neighbors                            
# since only 2 features, minkowski is same as euclidean distance                            
# where p=2 specifies sqrt(sum of squares). (p=1 is Manhattan distance)                         
knn = KNeighborsClassifier(n_neighbors=10,p=2,metric='minkowski')                           
knn.fit(X_train_std_knn,y_train_knn)                            
                            
y_pred_knn = knn.predict(X_test_std_knn)  # now try with the test data                          
                            
# combine the train and test data                           
# vstack puts first array above the second in a vertical stack                          
# hstack puts first array to left of the second in a horizontal stack                           
X_combined_std_knn = np.vstack((X_train_std_knn, X_test_std_knn))                           
y_combined_knn = np.hstack((y_train_knn, y_test_knn))                           
                            
# check results on combined data                            
y_combined_pred_knn = knn.predict(X_combined_std_knn)                           
                            
print('k-nearest Neighbor :')                           
​
k-nearest Neighbor :
# Note that this only counts the samples where the predicted value was wrong                            
print('Misclassified samples: %d' % (y_test_knn != y_pred_knn).sum())                           
                            
print('Accuracy: %.2f' % accuracy_score(y_test_knn, y_pred_knn))                            
print('Misclassified combined samples: %d' % (y_combined_knn != y_combined_pred_knn).sum())                         
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_knn, y_combined_pred_knn))                          
                            
cm  = confusion_matrix(y_test_knn, y_pred_knn)                          
precision = cm[0][0]/(cm[0][0]+cm[0][1])*100                            
print('Positive Predicted value:',str(precision))                           
#skplt.metrics.plot_confusion_matrix(y_test_knn, y_pred_knn,figsize=(8,8))                          
​
Misclassified samples: 1
Accuracy: 0.94
Misclassified combined samples: 2
Combined Accuracy: 0.97
Positive Predicted value: 100.0
​
​
​
​

