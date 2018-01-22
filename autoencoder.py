#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: pragadesh06
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable

#importing data

movies = pd.read_csv('ml-1m/movies.dat',sep = '::', 
                     header= None, engine='python', encoding = 'latin-1')

users = pd.read_csv('ml-1m/users.dat',sep = '::', 
                     header= None, engine='python', encoding = 'latin-1')

ratings = pd.read_csv('ml-1m/ratings.dat',sep = '::', 
                     header= None, engine='python', encoding = 'latin-1')

#preparing training and test data
training_set = pd.read_csv('ml-100k/u1.base',delimiter = '\t')
#converting to arrays
training_set = np.array(object=training_set, dtype = 'int') 

#test set
test_set = pd.read_csv('ml-100k/u1.test',delimiter = '\t')
#converting to arrays
test_set = np.array(object=test_set, dtype = 'int') 

#getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

#converting data into array with users in lines and movies in column
#creating list of list (943X1682)

#where 943 is the users and 1682 is the total movies.
#We need to map all the user with the movies, if there is no movie 
#rating by user, it is taken as Zero

#We find the list of movies and related rating by the user 
#create a new list of all movies with zeros and replace it with the users list



def convert(data):
    new_data =[]
    for id_users in range(1,nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

#converting training and test dataset to torch tensor

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#architecture of Auto Encoder
#inherited from torch nn module

class SAE(nn.Module):
    def __init__(self, ):#consider variables from module class from torch nn
        #super function to optimize the inheritence class
        super(SAE, self).__init__() #gets the all the functions and class from parent
        #first full connection from input to hidden layer
        self.fencode = nn.Linear(nb_movies, 20) #input is movies and no of hidden = 20
        #second full connection from first hidden layer to second
        self.sencode = nn.Linear(20, 10)
        #Decoding from hidden to output layer
        self.fdecode = nn.Linear(10, 20)
        #second decoding from 20 to no of movies
        self.sdecode = nn.Linear(20, nb_movies)
        #activation function
        self.activation = nn.Sigmoid()
    
    #forward propagation
    def forward(self, x):
        fst = self.activation(self.fencode(x))
        snd = self.activation(self.sencode(fst))
        thd = self.activation(self.fdecode(snd))
        frth = self.sdecode(thd) #no activation since it is the last layer in auto encoder
        return frth
    #end of the class modeule

#object initialization 
sae = SAE()

#criterian 
criteria = nn.MSELoss() #class inherited from torch nn class

#weight_decay reduces exploding gradient descent by reducing the learning rate after each convergence
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

#Training Stacked Auto Encoder
nb_epochs = 200
for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0.
    for id in range(nb_users):
        #no simple vector of one-dimention, so we take a fake dimension using variable
        input = Variable(training_set[id] ).unsqueeze(0)
        target = input.clone()
        #If the user have not rated a single or couple of movies, we neglect the ratings og that user
        if torch.sum(target.data > 0 ) > 0:
            #vector of predicted ratings
            output = sae(input)
            target.require_grad = False #to save up some memory by not computing on target
            output[target == 0] = 0
            loss = criteria(output, target)
            mean_corrected = nb_movies/ float(torch.sum(target.data > 0) + 1e-10) 
            loss.backward()
            train_loss += np.sqrt(loss.data[0] * mean_corrected)
            s += 1.
            optimizer.step() #intensity of the weight
    print('epoch:' + str(epoch) + 'loss:' + str(train_loss/s))
         
#Testing the SAE
test_loss = 0.
s =  0.
for id in range(nb_users):
    #no simple vector of one-dimention, so we take a fake dimension using variable
    input = Variable(training_set[id] ).unsqueeze(0)
    target = Variable(test_set[id] ).unsqueeze(0)
    #If the user have not rated a single or couple of movies, we neglect the ratings og that user
    if torch.sum(target.data > 0 ) > 0:
        #vector of predicted ratings
        output = sae(input)
        target.require_grad = False #to save up some memory by not computing on target
        output[target == 0] = 0
        loss = criteria(output, target)
        mean_corrected = nb_movies/ float(torch.sum(target.data > 0) + 1e-10) 
        test_loss += np.sqrt(loss.data[0] * mean_corrected)       
        s += 1.
print('loss:' + str(test_loss/s))        

    

        
        
        









