import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score

"""
PROBLEMAS A TENTAR RESOLVER:

Descritivo:



Predição:
*Predição de Venda
*Qual produto oferecer

"""


dataset = pd.read_csv('BlackFriday.csv')

print(dataset.columns)


dataset_categorical = dataset[['Product_Category_1', 'Product_Category_2']].copy()


#PREENCHENDO NULOS DA CATEGORIA 2
produto2_random = pd.DataFrame(dataset_categorical['Product_Category_2'].value_counts(1))
produto2_random = produto2_random.reset_index()
dataset_categorical['Product_Category_2'].fillna(np.random.choice(produto2_random.index, p=produto2_random.Product_Category_2), inplace=True)

#DIVIDINDO VARIÁVEIS POR CATEGORIA
categorical1 = dataset_categorical['Product_Category_1']
categorical2 = dataset_categorical['Product_Category_2']


#STANDARD SCALER
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


dataset_easy = dataset[['User_ID', 'Gender', 'Age', 'Occupation', 'City_Category','Stay_In_Current_City_Years', 'Marital_Status', 'Purchase']].copy()


#LABEL ENCODER
labelencoder = LabelEncoder()
dataset_easy['Gender'] = labelencoder.fit_transform(dataset_easy['Gender'])

labelencoder = LabelEncoder()
dataset_easy['Age'] = labelencoder.fit_transform(dataset_easy['Age'])

labelencoder = LabelEncoder()
dataset_easy['Occupation'] = labelencoder.fit_transform(dataset_easy['Occupation'])

labelencoder = LabelEncoder()
dataset_easy['City_Category'] = labelencoder.fit_transform(dataset_easy['City_Category'])

labelencoder = LabelEncoder()
dataset_easy['Stay_In_Current_City_Years'] = labelencoder.fit_transform(dataset_easy['Stay_In_Current_City_Years'])

labelencoder = LabelEncoder()
dataset_easy['Marital_Status'] = labelencoder.fit_transform(dataset_easy['Marital_Status'])

dataset_easy2 = dataset_easy[['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']].copy()


from info_gain import info_gain
"""
Gender  = info_gain.info_gain(dataset_easy2['Gender'], categorical1)
Age  = info_gain.info_gain(dataset_easy2['Age'], categorical1)
Occupation  = info_gain.info_gain(dataset_easy2['Occupation'], categorical1)
City_Category  = info_gain.info_gain(dataset_easy2['City_Category'], categorical1)
Stay_In_Current_City_Years  = info_gain.info_gain(dataset_easy2['Stay_In_Current_City_Years'], categorical1)
Marital_Status  = info_gain.info_gain(dataset_easy2['Marital_Status'], categorical1)
print(Gender, Age, Occupation, City_Category, Stay_In_Current_City_Years, Marital_Status)
"""
#Gender, Age e Occupation são os melhores preditores para prever categorical1

dataset_easy2 = dataset_easy2[['Gender', 'Age', 'Occupation']].copy()




dataset_easy = dataset_easy.groupby('User_ID')['Gender', 'Age', 'Occupation', 'City_Category',
       'Stay_In_Current_City_Years', 'Marital_Status', 'Purchase'].mean()



X = dataset_easy[['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']].copy()

y = np.array(dataset_easy['Purchase'].copy())



#ONEHOTENCODER
onehot = OneHotEncoder(categorical_features=[2, 4])
X = onehot.fit_transform(X).toarray()

onehot = OneHotEncoder(categorical_features=[2])
dataset_easy2 = onehot.fit_transform(dataset_easy2).toarray()



#Teste de Significância
"""
OLS = sm.OLS(exog=X, endog=y).fit()
print(OLS.summary())
"""



X = pd.DataFrame(X)
y = pd.Series(y)

X = X.drop([27, 29], axis=1)

#REMOVENDO OUTLIERS
max = y.quantile(0.75)+(y.quantile(0.75)-y.quantile(0.25))*1.5
min = y.quantile(0.25)+(y.quantile(0.75)-y.quantile(0.25))*1.5

for i in range(len(y)):
       if(y[i]>max or y[i]<min):
              y = y.drop(i, axis=0)
              X = X.drop(i, axis=0)





#DIVIDINDO O DATASET

from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(dataset_easy2, categorical1, test_size=0.2, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc2 = StandardScaler()
X_train2 = sc2.fit_transform(X_train2)
X_test2 = sc2.transform(X_test2)


#CLASSIFICADOR
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()

abc.fit(X_train2, y_train2)
y_pred_abc = abc.predict(X_test2)

print('y_pred_abc', y_pred_abc)

cm = confusion_matrix(y_test2, y_pred_abc)

print(pd.DataFrame(cm))


#      PREDIZENDO VALORES DOS PRODUTOS

#DIVIDINDO O DATASET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#ADABOOST
from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(learning_rate=0.1, loss='linear', n_estimators=900)
ada.fit(X_train, y_train)

y_pred_ada = ada.predict(X_test)



print('grid')
#GRID SEARCH
"""
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [300, 600, 900], 'learning_rate': [0.001, 0.01, 0.1], 'loss':['linear', 'square', 'exponential']}]
grid_search = GridSearchCV(estimator = ada,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10,
                           )

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print('best_accuracy = ', best_accuracy)
print('best_parameters = ', best_parameters)


y_pred_grid = grid_search.predict(X_test)

"""





#MÉTRICAS

print('RMSE/mean(y_test)', np.sqrt(mean_squared_error(y_test, y_pred_ada))/np.mean(y_test))
print('MAE/mean(y_test)', (mean_absolute_error(y_test, y_pred_ada)/np.mean(y_test)))
print('r2_score', r2_score(y_test, y_pred_ada))
