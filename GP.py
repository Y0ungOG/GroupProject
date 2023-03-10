import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier



#%% Intro to data
data = pd.read_csv('//ad.uillinois.edu/engr-ews/chenyim2/Downloads/MLF_GP1_CreditScore.csv')
data = data.drop('Rating',axis=1)
df = data
# Checking the shape of the dataset
print("Shape of the dataset:", data.shape)
print(data.head())

# Checking for missing values
print("Missing values in the dataset:\n", data.isnull().sum())

# Summary statistics of the dataset
print("Summary statistics:\n", data.describe())

#%% EDA
#correlation matrix heat map
corr = data.corr()
plt.figure(figsize=(12, 8))
plt.title("Correlation matrix")
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.xticks(np.arange(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(np.arange(len(corr.columns)), corr.columns)
plt.colorbar()
plt.show()

#Distribution of variables
plt.figure(figsize=(12, 8))
data.hist()
plt.show()

#scatter matrix(choose some of them as the dataset is too large)
ci = df.columns[12]
from pandas.plotting import scatter_matrix
cols = df[['Gross Margin', "EBITDA", 'Cash', 'ROA', 'InvGrd']]
scatter_matrix(cols, alpha=0.2, figsize=(12, 8), diagonal='kde')
plt.title('Scatter Plot')
plt.show()

#pair plot
sns.pairplot(df, vars=['Gross Margin', "EBITDA", 'Cash', 'ROA', 'InvGrd'])
plt.show()

#box plot
sns.boxplot(x='EBITDA', y='InvGrd', data=df)
plt.title('Box Plot')
plt.show()

#histogram
plt.hist(df['EBITDA'], bins=20)
plt.xlabel('EBITDA')
plt.ylabel('Frequency')
plt.title('Histogram of EBITDA')
plt.show()

#%%  Preprocessing feature extraction, feature selection
#Standardize data
scaler = StandardScaler()
X = df.drop('InvGrd',axis=1).values
y = df['InvGrd'].values
X = scaler.fit_transform(X)

#feature extraction by pca
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

#use feature selection based on random forest
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size = 0.1,random_state = 42)
feat_labels = df.columns[1:]

forest  = RandomForestClassifier(n_estimators = 200, random_state=42)
forest.fit(X_train,y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

#%% Model fitting and evaluation

#logisticregression
kfold = StratifiedKFold(n_splits = 10, random_state = 42, shuffle = True).split(X_train, y_train)
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=42))
scores = cross_val_score(estimator = pipe_lr, X = X_train, y = y_train, cv = 10, n_jobs =1)
print('Out of sample CV accuracy: %.3f +/- %.3f'%(np.mean(scores),np.std(scores)))

#decision tree
kfold = StratifiedKFold(n_splits = 10, random_state = 42, shuffle = True).split(X_train, y_train)
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), DecisionTreeClassifier(random_state=42))
scores = cross_val_score(estimator = pipe_lr, X = X_train, y = y_train, cv = 10, n_jobs =1)
print('Out of sample CV accuracy: %.3f +/- %.3f'%(np.mean(scores),np.std(scores)))

#random forest 
result_in = []
result_out = []
n_estimators_values = [5,25,50,100,200]
    #use pipeline to perform combined estimators
for n_estimator in n_estimators_values:
    pipe_lr = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimator), PCA(n_components=2))
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    in_sample_score = pipe_lr.score(X_train, y_train)
    out_of_sample_score = pipe_lr.score(X_test, y_test)
    result_in.append((n_estimator, in_sample_score))
    result_out.append((n_estimator, out_of_sample_score))
    
for results in result_in :
    print(f"Random state: {results[0]}, Score: {results[1]}")
for results in result_out :
    print(f"Random state: {results[0]}, Score: {results[1]}")

#calculate the mean 
result_df_in = pd.DataFrame(result_in)
mean_in = np.mean(result_df_in[1])  
print(mean_in)  
result_df_out = pd.DataFrame(result_out)
mean_out = np.mean(result_df_out[1])  
print(mean_out)  

#table
table1 = pd.DataFrame({
    'Metric':[ 'Mean in-sample train-test score', 'Std in-sample train-test score', 'Mean out-sample train-test score', 'Std out-sample train-test score'],
    'value':[mean_in, result_df_in[1].std(), mean_out, result_df_out[1].std()]})


#%% Hyperparameter tuning
#grid
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=42))
param_range = [0.01, 0.1, 1.0, 10.0]
param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


#random choose
param_dist = {
    'C': np.logspace(-3, 3, 7),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7))
}

# Create the SVC classifier
svc = SVC()

# Perform randomized search with cross-validation

search = RandomizedSearchCV(svc, param_distributions=param_dist, cv=5, n_iter=20, n_jobs=-1, random_state=42)
search.fit(X_train, y_train)

# Print the best hyperparameters and test score
print("Best parameters: ", search.best_params_)
print("Test score: {:.2f}".format(search.score(X_test, y_test)))


#%% Ensembling

#bagging
tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=None,
                              random_state=42)

bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500, 
                        max_samples=1.0, 
                        max_features=1.0, 
                        bootstrap=True, 
                        bootstrap_features=False, 
                        n_jobs=1, 
                        random_state=42)

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predictx(X_test)

bag_train = accuracy_score(y_train, y_train_pred) 
bag_test = accuracy_score(y_test, y_test_pred) 
print('Bagging train/test accuracies %.3f/%.3f'
      % (bag_train, bag_test))

#Ada boost
tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=1,
                              random_state=1)

ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=500, 
                         learning_rate=0.1,
                         random_state=1)

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

ada_train = accuracy_score(y_train, y_train_pred) 
ada_test = accuracy_score(y_test, y_test_pred) 
print('AdaBoost train/test accuracies %.3f/%.3f'
      % (ada_train, ada_test))

#voting
clf1 = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=1)
clf2 = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=1)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])

clf_labels = ['Logistic regression', 'Decision tree', 'KNN']

for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
    
mv_clf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('knn', clf3)], voting='soft')

clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]


for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))









