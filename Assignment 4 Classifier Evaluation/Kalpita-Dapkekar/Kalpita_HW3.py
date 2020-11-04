import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('cancer-data-train.csv',header = None,index_col = False)
#print(df)
X = df.iloc[:, :-1].values
y = df.iloc[:,-1].values
y[y == 'B'] = 0
y[y == 'M'] = 1
y = y.astype('int')
std_sclr = StandardScaler()
X = std_sclr.fit_transform(X)
C = [0.01, 0.1, 1, 10, 100]
arr = []
arr_mean = []
for val in C:
    SVM_clf = SVC(kernel='linear', C=val)
    scores = cross_val_score(SVM_clf, X, y, cv = 10,scoring='f1_macro')
    arr.append(scores)
    arr_mean.append(scores.mean())
print(arr_mean)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(C, arr_mean , color='darkblue', linewidth=2)
ax.scatter(C, arr_mean, color='darkgreen', marker='^')
ax.set_xlim(-0.10, 150)
ax.set_ylim(0.9, 1.0)
ax.set(title="Curve for SVM classifier", xlabel="C parameters", ylabel="F-measure in 10-fold cross validation")
plt.show()


criterion = {'gini','entropy'}
k = [2, 5, 10, 20]
for cri in criterion:
    arr1 = []
    arr1_mean = []
    for val in k:
        DTC = DecisionTreeClassifier(criterion = cri,max_leaf_nodes=val)
        scores = cross_val_score(DTC, X, y, cv = 10, scoring='f1_macro')
        arr1.append(scores)
        arr1_mean.append(scores.mean())
    print(arr1_mean)
    #print(arr1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k, arr1_mean, color='darkblue', linewidth=2)
    ax.scatter(k, arr1_mean, color='darkgreen', marker='^')
    ax.set(title=('Curve for', cri ,'criterion'), xlabel="Leaf nodes", ylabel="F-measure in 10-fold cross validation")
    ax.set_xlim(0.0, 25)
    ax.set_ylim(0.8, 1.0)
    plt.show()

df1 = pd.read_csv('cancer-data-test.csv',header = None,index_col = False)
X_test = df.iloc[:, :-1].values
y_test = df.iloc[:,-1].values
y_test[y_test == 'B'] = 0
y_test[y_test == 'M'] = 1
y_test = y_test.astype('int')
X_test = std_sclr.transform(X_test)

SVM_clf1 = SVC(kernel='linear', C=0.01)
SVM_clf1.fit(X,y)
y_pred_SVM = SVM_clf1.predict(X_test)
cm_SVM = confusion_matrix(y_test, y_pred_SVM)
print(cm_SVM)
print(classification_report(y_test, y_pred_SVM))
p_SVM = average_precision_score(y_test,y_pred_SVM)
R_SVM = recall_score(y_test,y_pred_SVM)
F_SVM = f1_score(y_test,y_pred_SVM)

DTC1 = DecisionTreeClassifier(criterion = 'gini',max_leaf_nodes=10)
DTC1.fit(X,y)
y_pred_DTC1 = DTC1.predict(X_test)
cm_DTC1 = confusion_matrix(y_test, y_pred_DTC1)
print(cm_DTC1)
print(classification_report(y_test, y_pred_DTC1))
p_DTC1 = average_precision_score(y_test,y_pred_DTC1)
R_DTC1 = recall_score(y_test,y_pred_DTC1)
F_DTC1 = f1_score(y_test,y_pred_DTC1)

DTC2 = DecisionTreeClassifier(criterion = 'entropy',max_leaf_nodes=20)
DTC2.fit(X,y)
y_pred_DTC2 = DTC2.predict(X_test)
cm_DTC2 = confusion_matrix(y_test, y_pred_DTC2)
print(cm_DTC2)
print(classification_report(y_test, y_pred_DTC2))
p_DTC2 = average_precision_score(y_test,y_pred_DTC2)
R_DTC2 = recall_score(y_test,y_pred_DTC2)
F_DTC2 = f1_score(y_test,y_pred_DTC2)

LDA = LinearDiscriminantAnalysis(n_components=1,store_covariance=True)
X_train = LDA.fit_transform(X, y)
y_pred_LDA = LDA.predict(X_test)
cm = confusion_matrix(y_test, y_pred_LDA)
print(cm)
print('Accuracy for LDA : ' + str(accuracy_score(y_test, y_pred_LDA)))
print(classification_report(y_test, y_pred_LDA))
p_LDA = average_precision_score(y_test,y_pred_LDA)
print('Precision of LDA',p_LDA)
R_LDA = recall_score(y_test,y_pred_LDA)
F_LDA = f1_score(y_test,y_pred_LDA)

RFC_classifier = RandomForestClassifier(n_estimators=100,max_depth=2, random_state=0)
RFC_classifier.fit(X, y)
y_pred_RFC = RFC_classifier.predict(X_test)
cm1 = confusion_matrix(y_test, y_pred_RFC)
print(cm1)
print('Accuracy for Random forest classifier : ' + str(accuracy_score(y_test, y_pred_RFC)))
print(classification_report(y_test, y_pred_RFC))
p_RFC = average_precision_score(y_test,y_pred_RFC)
R_RFC = recall_score(y_test,y_pred_RFC)
F_RFC = f1_score(y_test,y_pred_RFC)

objects = ('SVM', 'DT-gini', 'DT-entropy', 'LDA', 'RFC')
y_pos = np.arange(len(objects))
performance = [p_SVM,p_DTC1,p_DTC2,p_LDA,p_RFC]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Precision Score')
plt.title('Classifiers')
plt.show()

objects = ('SVM', 'DT-gini', 'DT-entropy', 'LDA', 'RFC')
y_pos = np.arange(len(objects))
performance = [R_SVM,R_DTC1,R_DTC2,R_LDA,R_RFC]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Recall Score')
plt.title('Classifiers')
plt.show()

objects = ('SVM', 'DT-gini', 'DT-entropy', 'LDA', 'RFC')
y_pos = np.arange(len(objects))
performance = [F_SVM,F_DTC1,F_DTC2,F_LDA,F_RFC]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('F_measure Score')
plt.title('Classifiers')
plt.show()