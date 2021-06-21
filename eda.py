#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Criação de algumas funções

#Função responsável por testar se a série é estacionária, basicamente ela plota a
# média móvel e a variância movel 
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = pd.Series(timeseries).rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
# %%
# Leitura da base de dados
data_base = pd.read_csv("CascadedTanksFiles/dataBenchmark.csv")
# %%
# Retirando as colunas que não ser analisadas
data_base.drop(['Ts', 'Unnamed: 5'],axis=1,inplace=True)
# %%
# Printando algumas informações iniciais, como: count, mean, std, min, max
data_base.describe()
#%%
# Criando uma variável timestamp vai de 1 até o tamanho da base de dados
time = [x for x in range(1,len(data_base[:])+1)]
# %%
#plotando uEst e uVal
sns.lineplot(data=data_base, x=time, y="uVal",color='red')
sns.lineplot(data=data_base, x=time, y="uEst", color='blue')
# %%
# Plotando yEst e yVal
sns.lineplot(data=data_base, x=time, y="yEst",color='red')
sns.lineplot(data=data_base, x=time, y="yVal", color='blue')
# %%
# Pelo gráfico é possível perceber que a série é estacionária
test_stationarity(data_base['uEst'])
#%%
# Gerando o gráfico de correlação entre o uEst e yEst
sns.heatmap(data_base.corr(), annot = True)