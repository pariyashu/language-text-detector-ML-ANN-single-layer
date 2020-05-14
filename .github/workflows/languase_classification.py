# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:05:04 2020

@author: Paribartan Humagain
"""
import os
import glob
import numpy as np

# make list of the paths to the each input languase test
read_files_english = glob.glob(os.path.join("C:/Users/pariy/OneDrive/Desktop/suman_project/Single_layer_ANN/English","*.txt"))
read_files_german = glob.glob(os.path.join("C:/Users/pariy/OneDrive/Desktop/suman_project/Single_layer_ANN/German","*.txt"))
read_files_polish = glob.glob(os.path.join("C:/Users/pariy/OneDrive/Desktop/suman_project/Single_layer_ANN/Polish","*.txt"))

# combine all the paths to one list 
total_list_read_files =  read_files_english + read_files_german + read_files_polish 
#create empty list with 26 elements
empty_list = [0]*26

# define function to count each alphabet occurance in the text
def count_the_variable(filename,asci_value):
    count = 0
    # 97 becasus we want to check a -z if they are im capital we will convert them first 
    asci_number = 97+asci_value
    for i in range(26):
  # find the total number ofeach character in the text and put the value in the list
        with open(filename,encoding='utf-8') as f:
            count = 0
            for line in f:
                for char in line:
                    if char.isalpha():
                        if char.isupper():
                            char = char.lower()
                        if char == chr(asci_number):
                            count += 1
                            
            return count 
        
# define a function to convert the occurance of all latin alphabet of all texts to an array
def convert_text_to_vec(x):
    input_vector = np.array([[0]*26])
    for k in range(len(x)):
        for i in range(26):
            empty_list[i] = count_the_variable(x[k],i)
        #normolize the vector 
        for i in range(26):
            empty_list[i] = empty_list[i]/sum(empty_list)
            
        input_i = np.array([empty_list])
        input_vector = np.concatenate((input_vector, input_i), axis=0)
    X= np.array(input_vector)
    X = np.delete(X, (0), axis=0)
    return(X)
    

      
# finalize the input and output for the training 
X =  convert_text_to_vec(total_list_read_files)

# give the node value [0,0,1] to English, [0,1,0] to german and [1,0,0] to polish for encoding

y = np.array(([0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],
            [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
            [1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]),
           dtype=float)
    
    

# Create a neural net

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)

# Class definition
class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x
        #initialize the weights of the edge connecting input vec and hidden layer i.e 26*4 = 104
        self.weights1= np.random.rand(self.input.shape[1],4) # considering we have 4 nodes in the hidden layer
        #initialize the weights of the edge connecting hidden layer and the output layer i.e 4*3 = 12
        self.weights2 = np.random.rand(4,3)
        self.y = y
        self.output = np. zeros(y.shape)
        
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2
        
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
    
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        
        self.backprop()

    def predict(self, x):
        self.input = x
        out = self.feedforward()
        return out
        
        
# build and train the neural network 
NN = NeuralNetwork(X,y)

for i in range(1500): # trains the NN 1,500 times

    NN.train(X, y)
    
# take input from user 
user_input=input(str('please provide the path of the text that you want program to predict:       '))
read_user_input=glob.glob(os.path.join(user_input,"*.txt"))

# convert the text to occurance of latin alphabet vector for input
test_input = convert_text_to_vec(read_user_input)

# predict the user input
prediction = NN.predict(test_input)

#convert to the absolut values with activation threshold 0.5
prediction = (prediction>0.5).astype(int)

# provide the text output mentioning the languase 
if (prediction == [[0,0,1]]).all():
    print('your given text is in English language')
elif (prediction == [[0,1,0]]).all():
    print('your given text is in German language')
elif (prediction == [[1,0,0]]).all():
    print('your given text is in Polish language')
else:
    print('sory could not predict may be I am a stupid Machine!! :( ')

