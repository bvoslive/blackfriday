import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

dataset = pd.read_csv('BlackFriday.csv')

#ELIMINANDO VARIÁVEIS DESNECESSÁRIAS
dataset = dataset.drop(['User_ID', 'Product_Category_2', 'Product_Category_3', 'Product_ID'], axis=1)

#FAZENDO UMA AMOSTRA
dataset = dataset.sample(int(len(dataset)*0.2))

#SETANDO INDEXES E COLUNAS
dataset.index = range(len(dataset))
colunas = dataset.columns
dataset.columns = range(len(dataset.columns))

#GRÁFICO DE VARIÁVEIS CATEGÓRICAS
for i in range(len(dataset.columns)):
    if(dataset[i].describe().dtype=='object'):
        sns.boxplot(dataset[i], dataset[7])
        plt.title(colunas[i])
        plt.show()


#ELIMINANDO OUTLIERS
quartil3 = dataset[7].quantile(0.95)

for i in range(len(dataset)):
    if(dataset[7][i]>quartil3):
        dataset = dataset.drop(i)



X = dataset.iloc[:, [0,1,2,3,4,5,6]].values
y = dataset.iloc[:, 7].values

#NORMALIZANDO Y
y = np.log(y)


#LABEL ENCODER
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])

labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

labelencoder = LabelEncoder()
X[:, 4] = labelencoder.fit_transform(X[:, 4])


X = np.array(X, dtype=float)

#ONEHOT ENCODER
onehot = OneHotEncoder(categorical_features=[4, 6])
X = onehot.fit_transform(X).toarray()


#VERIFICANDO VALOR P
regressor_OLS = sm.OLS(exog=X, endog=y).fit()
print(regressor_OLS.summary())

#TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=300)
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)


y_test = np.exp(y_test)
y_pred = np.exp(y_pred)


#MÉTRICAS
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("RMSE", mean_squared_error(y_test, y_pred)**0.5)
print("mean_absolute_error", mean_absolute_error(y_test, y_pred)/np.mean(y_test))
