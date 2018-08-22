# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 20:38:44 2018

@author: Romulo

Essa apliação já estar com signoid

"""
from random import random
import numpy as np
import pandas as pd
#importação para dividir treino de teste
from sklearn.model_selection import train_test_split
#importação para Normalizar e Padronização
from sklearn.preprocessing import MinMaxScaler
vetorErro_treino=[]
vetorErro_teste=[]
vetor_saida =[]

baseIris=pd.read_csv('work.csv')
entrada=baseIris.iloc[:,0:2].values
classe=baseIris.iloc[:,2]

#separar as classes em vetores
saidas=[]
for i in classe:
        if i== 0:
            saidas.append([1,0,0])       
        elif i== 1:
            saidas.append([0,1,0])
        else:
            saidas.append([0,0,1])

saidas=np.array(saidas)

scaler = MinMaxScaler()
scaler.fit(entrada)
entradaNorma=scaler.transform(entrada)

    
def Modificar(matriz):
        
        for linha in matriz:
            maximo = np.amax(linha)         
            for i in range(3):
                if linha[i] >= maximo:
                    linha[i] = 1
                else:
                    linha[i] = 0
        return matriz
        
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)


def vetor(mat):
        for i in mat:
            
                if i[0] == 1:
                    vetor_saida.append(0)       
                elif i[1] == 1:
                    vetor_saida.append(1)
                elif i[2] == 1:
                    vetor_saida.append(2)
                else:
                    vetor_saida.append(random.range(0,3))
                    
        return vetor_saida

epoca=10000
taxa_aprendizado=0.0035
momento=1
pesos=np.random.randn(2,3)

for i in range(20):
    for i in range(epoca):
        
        X_treino, X_teste, Y_treino, Y_teste =train_test_split(entrada,saidas) 
        
        #treino
        somaSinapse=np.dot(X_treino, pesos) 
        Camada_Saida=sigmoid(somaSinapse)
        camada =np.copy(Camada_Saida)
        Modificar(camada) 
        
        erroCamadaSaidas=Y_treino-camada
        mediaAbsoulta=np.mean(np.abs(erroCamadaSaidas))
        vetorErro_treino.append(mediaAbsoulta)
        
        derivadaSaida = sigmoidDerivada(Camada_Saida)
        deltaSaida = (erroCamadaSaidas*derivadaSaida).T
        pesosNovos = (deltaSaida.dot(X_treino)).T
        pesos = (pesos)+(pesosNovos*taxa_aprendizado)
        
        #teste   
        somaSinapse_teste=np.dot(X_teste, pesos) 
        Camada_Saida_teste=sigmoid(somaSinapse_teste)
        camada_teste =np.copy(Camada_Saida_teste)
        Modificar(camada_teste) 
        
        erroCamadaSaidas_teste=Y_teste-camada_teste
        mediaAbsoulta_teste=np.mean(np.abs(erroCamadaSaidas_teste))
        vetorErro_teste.append(mediaAbsoulta_teste)
        
    
    print("Erro Treino: "+str(mediaAbsoulta))
    print("Erro Teste: "+str(mediaAbsoulta_teste)) 

erro_treino=np.array(vetorErro_treino)
erro_teste=np.array(vetorErro_teste)


media_treino=np.mean(erro_treino)
media_teste=np.mean(erro_teste)

print('')
print("Erro para o Treino: "+str(media_treino))  
print("Erro para o Teste: "+str(media_teste)) 

vet = vetor(camada)
v = np.array(vet)


from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.svm import SVC

#X = X_teste[:, [0, 2]]
#y=v_1

X = X_treino[:, [0, 1]]
y=v

# Training a classifier
svm = SVC(C=0.1, kernel='linear')
svm.fit(X, y)

# Plotting decision regions
plot_decision_regions(X, y, clf=svm, legend=2)

# Adding axes annotations
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.title('Perceptron')
plt.show()
