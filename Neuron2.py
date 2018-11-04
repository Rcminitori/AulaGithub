# -*- coding: utf-8 -*-
"""
Auor: Romulo Cesar
03 de Junho de 2018
Trabalho_01 de Redes Neurais.
"""
import random
import numpy as np
from sklearn.metrics import mean_squared_error


class Ajustes:
    
     def __init__(self,entrada, saida,taxa_aprendizado = 0.0002, biasErro = 0.000001, bias = 1, epoca=10000):
         self._taxa_aprendizado = taxa_aprendizado
         #Limite representando o viés (b da equação y = ax + b)
         self._biasErro = biasErro
         self._epoca=epoca
         self._bias = bias
         self._entrada=entrada
         self._saida = saida   
         #Randomise pesos (Coloque mais uma entrada para o vetor de pesos para o viés).
         self._W_inicio =[]
         self._Y=np.array(self._W_inicio)
         self._global_e = 1.0
     
     def  Ajuste(self):
         
         self._entrada=np.insert(self._entrada, 0,self._bias, axis=1)
        
         #self._saida=np.insert(saida, len(self._saida)-1,self._bias)
         self._W_inicio = list( map(lambda x: x*random.random(), self._entrada))
         self._W=np.array(self._W_inicio)
           
         #Calculo Saida.
         Epoca = 0
         EqmW = 0
         for i in range(20):
             while True:
                Eqmanterior = EqmW
               
                u=(self._W*self._entrada)
                
                for i in range(len(self._saida)):
                    a=np.sum(u[i])
                    self._Y=np.insert(self._Y,i,a)
                    #self._Y=np.broadcast(a)
                    #print(self._Y)
                    #verificar a diferença das saídas de redes e deseja
                    if a!=self._saida[i]:
                        erro = self._saida[i] - a
                        self._W[i]=self._W[i]+self._taxa_aprendizado*erro*self._entrada[i]
               
          
                MSE=mean_squared_error(self._saida,self._Y)
                RMSE=np.sqrt(MSE)
                #self._Y=self._Y.reset()
                y=[]
                self._Y=np.array(y)
                Epoca += 1
                EqmW =MSE/len(self._entrada) 
                Eqmatual = EqmW
                #print(abs(Eqmatual-Eqmanterior))
                if  abs(Eqmatual-Eqmanterior) <=self._biasErro or Epoca>=self._epoca:
                       
                    break
             print("MSE:{}".format(MSE))
             print("RMSE:{}".format(RMSE))   
         print("Épocas:{}".format(Epoca))
         
     def Pesos(self):
         return self._W
            

               
   
   