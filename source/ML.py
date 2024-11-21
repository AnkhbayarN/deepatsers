import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import CalibratedClassifierCV

from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import GridSearchCV


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.metrics import specificity_score
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import numpy as np

def _importData(file_name, classification, columns):
    ds = pd.read_csv(file_name, low_memory=False)
    y = ds[classification]
    x = ds.drop(columns, axis = 1)
    # x = x.iloc[:, ][['1078', '1582']]
    return x, y

def _createModel(mdl):
    if mdl == 'svm':
        # model = CalibratedClassifierCV(SVC(kernel = 'linear', C=2.0))
        param_grid = {
            'C': [0.1, 1, 4, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1]
        }
        model = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)

    elif mdl == 'gbm':
        model = GradientBoostingClassifier(n_estimators=500, max_depth=300, learning_rate=0.01,random_state=100,max_features=5 )
        # model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,random_state=100,max_features='sqrt')
    elif mdl == 'knn':
        model = KNeighborsClassifier(n_neighbors=1) #1, 6   
    elif mdl == 'rf':
        model = RandomForestClassifier(n_estimators = 300, max_depth = 300)  
    return model

def _fit(model, x_train, y_train):
    model.fit(x_train, y_train)

def _predict(model, x_test):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)
    return y_pred, y_pred_proba

def _conf_matrix(labels, title, y_test, y_pred, multi_or_binary, epsname):
    cm = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_title(title)
    plt.rcParams.update({'font.size':18})
    if (multi_or_binary == 'binary'):
        cmd.plot(ax=ax, cmap=plt.cm.Blues, values_format='g')
    else:
        cmd.plot(xticks_rotation=90, ax=ax, cmap=plt.cm.Blues, values_format='g')

def _report(labels, y_test, y_pred):
    print(classification_report(y_test, y_pred, target_names=labels, digits=4))

def _auc(y_test,y_pred_proba):
    auc = metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    return auc

def _specificity(y_test, y_pred):
    print('specificity:', specificity_score(y_test, y_pred, average='weighted'))

mdl = 'rf' #rf, gbm, svm, knn

file_train = 'cv_folds/train_fold_1.csv'
file_test = 'cv_folds/test_fold_1.csv'
columns = ['multiclassification', 'multiname', 'binaryclassification', 'binaryname', 'realorgan']

classification = 'multiclassification'
labels = ['Blank', 'Streptavidin', 'VLP', 'S', 'N']

title = mdl.upper() + '(all peak)'
epsname = mdl + '_real_allpeak_pc'

x, y = _importData(file_train, classification, columns)

model = _createModel(mdl)
_fit(model, x, y)
x_test, y_test = _importData(file_test, classification, columns)
y_pred, y_pred_proba = _predict(model, x_test)

_report(labels, y_test, y_pred)
_conf_matrix(labels, title, y_test, y_pred, 'multi', epsname)
_specificity(y_test, y_pred)
print('auc:' + str(_auc(y_test, y_pred)))