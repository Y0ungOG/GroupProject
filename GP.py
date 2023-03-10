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


#%% Intro to data
data = pd.read_csv('//ad.uillinois.edu/engr-ews/chenyim2/Desktop/ccdefault.csv')
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
ci = df.columns[1]
from pandas.plotting import scatter_matrix
cols = df[['LIMIT_BAL', "AGE", 'PAY_3', 'DEFAULT', 'BILL_AMT1']]
scatter_matrix(cols, alpha=0.2, figsize=(12, 8), diagonal='kde')
plt.title('Scatter Plot')
plt.show()

#pair plot
sns.pairplot(df, vars=['LIMIT_BAL', "AGE", 'PAY_3', 'DEFAULT', 'BILL_AMT1'])
plt.show()

#box plot
sns.boxplot(x='LIMIT_BAL', y='BILL_AMT1', data=df)
plt.title('Box Plot')
plt.show()

#histogram
plt.hist(df['AGE'], bins=20)
plt.xlabel('AGE')
plt.ylabel('Frequency')
plt.title('Histogram of AGE')
plt.show()

#%%  Preprocessing feature extraction, feature selection
#Standardize data
scaler = StandardScaler()
x = df.iloc[:,:-1].values
y = df['DEFAULT'].values
x = scaler.fit_transform(x)

#feature extraction by pca
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)
df_pca = pd.DataFrame(data=x_pca, columns=['PC1', 'PC2'])

#use feature selection based on random forest
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state = 42)
feat_labels = df.columns[1:]

forest  = RandomForestClassifier(n_estimators = 200, random_state=42)
forest.fit(x_train,y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(x_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(x_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.tight_layout()
plt.show()

#%% Model fitting and evaluation

#logisticregression
kfold = StratifiedKFold(n_splits = 10, random_state = 42, shuffle = True).split(x_train, y_train)
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=42))
scores = cross_val_score(estimator = pipe_lr, X = x_train, y = y_train, cv = 10, n_jobs =1)
print('Out of sample CV accuracy: %.3f +/- %.3f'%(np.mean(scores),np.std(scores)))

#decision tree
kfold = StratifiedKFold(n_splits = 10, random_state = 42, shuffle = True).split(x_train, y_train)
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), DecisionTreeClassifier(random_state=42))
scores = cross_val_score(estimator = pipe_lr, X = x_train, y = y_train, cv = 10, n_jobs =1)
print('Out of sample CV accuracy: %.3f +/- %.3f'%(np.mean(scores),np.std(scores)))

#random forest 
result_in = []
result_out = []
n_estimators_values = [5,25,50,100,200]
    #use pipeline to perform combined estimators
for n_estimator in n_estimators_values:
    pipe_lr = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimator))
    pipe_lr.fit(x_train, y_train)
    y_pred = pipe_lr.predict(x_test)
    in_sample_score = pipe_lr.score(x_train, y_train)
    out_of_sample_score = pipe_lr.score(x_test, y_test)
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

gs = gs.fit(x_train, y_train)
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
search.fit(x_train, y_train)

# Print the best hyperparameters and test score
print("Best parameters: ", search.best_params_)
print("Test score: {:.2f}".format(search.score(x_test, y_test)))


#%% Ensembling
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

















