#%% 
# Importando as bibliotecas
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from matplotlib import pyplot as plt
import seaborn as sns
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# %%
# Leitura da base de dados
data_base = pd.read_csv("CascadedTanksFiles/dataBenchmark.csv")
#%%
# capturando as variáveis de entrada e saída de treinamento e validação
x_train = data_base["uEst"]
y_train = data_base["yEst"]
x_val   = data_base["uVal"]
y_val   = data_base["yVal"]
#%%
# convertando em numpy array e dando um reshape nas mesmas
x_train, y_train= np.array(x_train), np.array(y_train)
x_val, y_val= np.array(x_val), np.array(y_val)
x_train, y_train= x_train.reshape((1024,1)), y_train.reshape((1024,1))
x_val, y_val= x_val.reshape((1024,1)), y_val.reshape((1024,1))
# %%
# vamos gerar os numpy array. o primeiro vai conter as 10 leituras da varíavel x 
#. e o segundo vai conter a próxima leitura da variável yEst
previsores = []
valor_real = []

for i in range(10, (np.size(x_train[:,0])-10)):
    # pega leituras 10 anteriores 
    previsores.append(x_train[i-10:i,])
    # pega a 11 amostra 
    valor_real.append(y_train[i,0])

#%%
# Convertendo em numpy.array
previsores, valor_real = np.array(previsores), np.array(valor_real)
# %%
regressor = Sequential()
# Vamos acrescentar uma primeira camada LSTM com 30 camadas "enroladas" na camada escondida
# o parâmetro return_sequences significa que ele vai passar o resultado para frente  para as próximas camadas
# no input_shape dizemos como é a nossa entrada. temos seis entradas atrasadas ou amostrada em 20 segundos
# return_sequences retornam a saída do estado oculto para cada etapa do tempo de entrada.
regressor.add(LSTM(units = 30, input_shape = (previsores.shape[1],1)))
# Vamos criar a camada de saída 
regressor.add(Dense(units = 1, activation = 'linear'))
# Vamos compilar a rede
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])
# função early stop vai para de treinar a rede se algum parâmetro monitorado parou de melhorar
es = EarlyStopping(monitor ='loss', min_delta = 1e-10, patience = 10, verbose = 1)
# ele vai reduzir a taxa de aprendizagem quando uma metrica parou de melhorar
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
mcp =  ModelCheckpoint(filepath=r'pesos_lstm.h5', monitor = 'loss', save_best_only= True)
history = regressor.fit(previsores, valor_real, epochs = 200, batch_size = 30, callbacks = [es,rlr,mcp])
regressor_json = regressor.to_json()
# %%
print(history.history.keys())
print(history.history['loss'])
# %%
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#%%
entradas = []
saidas   = []
# esse for vai criar as matrizes de três dimensões. Dos dados de entrada de teste
# E os valores reais dos dados de saída para ser comparado com os valores previstos pela rede
for i in range(10, (np.size(x_val[:,0])-10)):
    # pega 20 anteriores
    entradas.append(x_val[i-10:i,:])
    # pega as 21:25 para ser a saída correspondente as 20 leituras anteriores 
    saidas.append(y_val[i,0])
# %%
entradas, saidas = np.array(entradas), np.array(saidas)
#%%
previsao = regressor.predict(entradas)
# %%
sns.lineplot(data=previsao[:,0],color='red')
sns.lineplot(data=saidas, color='blue')
plt.legend(labels=['Valor previsto', 'Valor Real'])
# %%

# %%
