import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score

"""
PROBLEMAS A TENTAR RESOLVER:

Descritivo:



Predição:
*Predição de Venda
*Qual produto oferecer

"""

#IMPORTANDO VARIÁVEIS
dataset = pd.read_csv('C:/Users/Bruno/PycharmProjects/untitled1/_Kaggle/blackfriday/BlackFriday.csv')
purchase = dataset['Purchase']


#COLUNAS DO DATASET
print('Colunas', dataset.columns)


#   ---TRATAMENTO DOS DADOS---

#CATEGORIA DE CADA PRODUTO
dataset_categorical = dataset[['Product_ID', 'Product_Category_1', 'Product_Category_2', 'Purchase']].copy()


#PREENCHENDO NULOS DA COLUNA CATEGORIA 2
produto2_random = pd.DataFrame(dataset_categorical['Product_Category_2'].value_counts(1))
produto2_random = produto2_random.reset_index()
dataset_categorical['Product_Category_2'].fillna(np.random.choice(produto2_random.index, p=produto2_random.Product_Category_2), inplace=True)
dataset_categorical['Product_Category_2'] = (dataset_categorical['Product_Category_2']).astype(int)


#AGRUPANDO DATASET

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#LABEL ENCODER
labelencoder1 = LabelEncoder()
dataset['Gender'] = labelencoder1.fit_transform(dataset['Gender'])

labelencoder2 = LabelEncoder()
dataset['Age'] = labelencoder2.fit_transform(dataset['Age'])

labelencoder3 = LabelEncoder()
dataset['Occupation'] = labelencoder3.fit_transform(dataset['Occupation'])

labelencoder4 = LabelEncoder()
dataset['City_Category'] = labelencoder4.fit_transform(dataset['City_Category'])

labelencoder5 = LabelEncoder()
dataset['Stay_In_Current_City_Years'] = labelencoder5.fit_transform(dataset['Stay_In_Current_City_Years'])

labelencoder6 = LabelEncoder()
dataset['Marital_Status'] = labelencoder6.fit_transform(dataset['Marital_Status'])


dataset_easy = dataset.groupby('User_ID', as_index=False)['Gender', 'Age', 'Occupation', 'City_Category',
       'Stay_In_Current_City_Years', 'Marital_Status', 'Purchase'].mean()

dataset_easy['Gender'] = labelencoder1.inverse_transform(dataset_easy['Gender'].astype(int))
dataset_easy['Age'] = labelencoder2.inverse_transform(dataset_easy['Age'].astype(int))
dataset_easy['Occupation'] = labelencoder3.inverse_transform(dataset_easy['Occupation'].astype(int))
dataset_easy['City_Category'] = labelencoder4.inverse_transform(dataset_easy['City_Category'].astype(int))
dataset_easy['Stay_In_Current_City_Years'] = labelencoder5.inverse_transform(dataset_easy['Stay_In_Current_City_Years'].astype(int))
dataset_easy['Marital_Status'] = labelencoder6.inverse_transform(dataset_easy['Marital_Status'].astype(int))


#SETANDO DADOS DA VARIÁVEL MARITAL_STATUS COMO BOOLEAN
dataset_easy['Marital_Status'] = dataset_easy['Marital_Status'].astype(bool)


dataset_easy.to_excel('C:/Users/Bruno/PycharmProjects/untitled1/_Kaggle/blackfriday/output.xlsx')


#   ---INSIGHTS---



#RECOMENDA PRODUTO A PARTIR DA CATEGORIA 1

vetor = []

for i in range(1, len((dataset_categorical['Product_Category_1'].unique()))):

    k = dataset_categorical['Product_Category_2'][dataset_categorical['Product_Category_1']==i]
    k=k.value_counts(1)

    vetor.append([i, k.index, k.values])


insere = int(input())

for i in range(len(vetor)):
    if(vetor[i][0]==insere):
        df = pd.DataFrame({'Produto':vetor[i][1], 'Probabilidade':(vetor[i][2]*100).round(2)})
        print(df)





for i in range(len(dataset_categorical)):
    vetor.append(dataset_categorical['Product_Category_1'][dataset_categorical['Product_Category_1']==1])



#CATEGORIA DE PRODUTOS
def cat_produtos():
    prod1 = dataset_categorical['Product_Category_1'].value_counts()
    prod2 = dataset_categorical['Product_Category_2'].value_counts()

    prods = pd.DataFrame({'Cat Produto 1': prod1, 'Cat Produto 2':prod2})
    prods.plot.bar()
    plt.title('Contagem por categorias de produtos2')
    plt.show()

cat_produtos()




#COMPRA_IDADE_GÊNERO
def compra_idade_genero():
    sns.barplot(dataset_easy['Age'], dataset_easy['Purchase'], hue=dataset_easy['Gender'], order=sorted(dataset_easy['Age'].unique()))
    plt.show()






#DESCRIÇÃO DAS VARIÁVEIS
def descricao():
    colunas = dataset_easy.columns
    dataset_describe = dataset_easy.copy()
    dataset_describe.columns = range(len(dataset_describe.columns))

    for i in range(len(dataset_describe.columns)):
        if(dataset_describe[i].describe().dtype=='object'):
            print('---',str.upper(str(colunas[i])),'---\n')
            print(dataset_describe[i].describe())
            print('Valores em porcento: \n', dataset_describe[i].value_counts(2))


descricao()

#BOXPOLOT CIDADE X GÊNERO X PURCHASE
def boxplot_cidade_genero():
    sns.boxplot(dataset_easy['City_Category'], dataset_easy['Purchase'], hue=dataset_easy['Gender'], order=sorted(dataset_easy['City_Category'].unique()))
    plt.show()
    """Nota-se que há maior poder aquisitivo na Cidade C e está maior entre os homens
        porém poucos indívuduos do sexo masculino acrescentaram valor no gráfico da
        Cidade C, visto que mediana destes indivíduos está próxima da categoria
        do sexo feminino"""

boxplot_cidade_genero()

#HISTOGRAMA ENTRE HOMENS E MULHERES
def histograma_homens_e_mulheres():
    homem_compra = dataset_easy['Purchase'][dataset_easy['Gender']=='M']
    mulher_compra = dataset_easy['Purchase'][dataset_easy['Gender']=='F']
    #homem_compra = np.log(homem_compra)
    #mulher_compra = np.log(mulher_compra)

    print('Curtose - Homem', homem_compra.kurtosis())
    print('Curtose - Mulher', mulher_compra.kurtosis())

    print('homens somatório', homem_compra.sum())
    print('mulher somatório', mulher_compra.sum())

    print('homens contagem', len(homem_compra))
    print('mulher contagem', len(mulher_compra))

    print('Comparação entre homens e mulheres - Número das compras', round(len(homem_compra)/(len(homem_compra)+len(mulher_compra)), 3))
    print('Comparação entre homens e mulheres - Valor de compras', round(homem_compra.sum() / (homem_compra.sum() + mulher_compra.sum()), 3))



    sns.distplot(homem_compra, label='Homem')
    sns.distplot(mulher_compra, label='Mulher')
    plt.legend()
    plt.show()

histograma_homens_e_mulheres()


#HISTOGRAMA ENTRE CASADOS E NÃO CASADOS
def histograma_casados():
    casado = dataset_easy['Purchase'][dataset_easy['Marital_Status']==True]
    nao_casado = dataset_easy['Purchase'][dataset_easy['Marital_Status']==False]
    #homem_compra = np.log(homem_compra)
    #mulher_compra = np.log(mulher_compra)

    print('Curtose - Homem', casado.kurtosis())
    print('Curtose - Mulher', nao_casado.kurtosis())

    print('homens somatório', casado.sum())
    print('mulher somatório', nao_casado.sum())

    print('homens contagem', len(casado))
    print('mulher contagem', len(nao_casado))

    print('Comparação entre homens e mulheres - Número das compras', round(len(casado)/(len(casado)+len(nao_casado)), 3))
    print('Comparação entre homens e mulheres - Valor de compras', round(casado.sum() / (casado.sum() + nao_casado.sum()), 3))

    sns.distplot(casado, label='Casado')
    sns.distplot(nao_casado, label='Não-Casado')
    plt.legend()
    plt.show()


histograma_casados()


"""
Homens compram produtos mais caros
"""
