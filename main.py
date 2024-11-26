import torch
import torch.nn as nn
import torch.nn.functional as func

import matplotlib.pyplot as plt




#Class 1: nn.Module, which is base class for all neural network modules 
class Model(nn.Module):
    #Input layer, receives "raw data" from the outside world, sends to hidden layers "black box"
    def __init__(self, in_features = 4, h1 = 8, h2 = 9, out_features = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1) #using 4 input features, creates a linear (dense) layer with 8 neurons
        self.fc2 = nn.Linear(h1, h2) #using 8 input neurons, creates a second linear (dense) layer with 9 neurons
        self.out = nn.Linear(h2, out_features) #creates an output layer that will connect with

    #Take X_test data and push it through layers to produce output
    def forward(self, x): #x is the input tensor, which is the data fed into the model for processing
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.out(x)

        return x

torch.manual_seed(41)
model = Model()


#import iris dataset from sklearn
#Convert this dataset to pandas framework
from sklearn import datasets
iris = datasets.load_iris()
import pandas as pd
#print(type(iris))
#print(iris)
#print(iris.target_names)
dataframe = pd.DataFrame(data = iris.data, columns = iris.feature_names)
dataframe['target'] = iris.target

#Now split imported data into Training set (X) and Testing set (y)
X = dataframe.drop('target', axis = 1) #Drops the first column of DataFrame
y = dataframe['target']

#Now convert X and y to numpy arrays
import numpy as np
X = X.values
y = y.values

from sklearn.model_selection import train_test_split
#Splits data into random train and test subsets
#20 percent of data is used for test, 80 percent is for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 37) 

#Now use Tensor to batch 2d (data and feature names) data into 3d 
#Converts X features to float tensors, float for decimals
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
#Converts y labels to long tensors, long for integers
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Define loss function
criterion = nn.CrossEntropyLoss()
#Adaptive Movement Estimation Adam, keeps track of gradient moving average (momentum)
#and squared gradient moving average (adaptive rates)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) #returns iterable containing updated weights


#Model Training
#Each Epoch is one run through all training data
epochs = 200
#Keep track of loss (goal is to minimize this through training!)
loss = []
for i in range(epochs):
    y_pred = model.forward(X_train) #Get predicted results

    #Measure and keeps track of loss
    losses = criterion(y_pred, y_train) #predicted value vs the y_train value
    loss.append(losses.detach().numpy())

    #Prints the updated loss array every 10 epoch
    if i % 10 == 0:
        print('Epoch: ' +str(i) + ' and loss:' + str(loss[-1]))
        print()

    #Gradient descent
    #Minimizes predicted and actual results
    optimizer.zero_grad() #Resets gradient of all optimized torch
    losses.backward() #Calculates gradient of loss function
    optimizer.step() #Updates weights to move against gradient of loss function


#Graph loss with matplotlib
plt.plot(range(epochs), loss, label = "Loss")
plt.ylabel("loss/error")
plt.xlabel("epoch")
plt.title("Loss over Epochs")
plt.show()


################################################
#Now we evaluate model on test data set!

#1) Disable gradient computation 
with torch.no_grad():
   #Call the forward function
    y_eval = model.forward(X_test)
    test_loss = criterion(y_eval, y_test) #Calculates loss
    print(test_loss)


#Write a function that outputs name associated with the prediction value
def name_output(tensor: torch.Tensor):
    name = ""
    name_index = -1
    if tensor.argmax(y_val).item() == 0:
        name = "Setosa"
        name_index = 0
    elif tensor.argmax(y_val).item() == 1:
        name = "Versicolor"
        name_index = 1
    else:
        name = "Virgini"
        name_index = 2
    return name, name_index

#Now evaluate the test set and output results
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        # Prints out rows of three values
        # Associated with Setosa(0), Versicolor(1), and Virgini(2) correspondingly
        # To the right, print out the correct answer
        name, name_index = name_output(torch)
        print(str(i+1) + ". " + str(y_val) + "; Highest value: " + str(y_test[i].item()) + "; It is a " + name) 
        #See if the predictions are correct or not
        if y_val.argmax().item() == y_test[i]:
            correct += 1
    print("We got " + str(correct) + " correct!")






######################################################
#Now Evaluate New Data on the Model!

#Create a new data point
#Imagine we have a new flower, and we made the following measurements:
#Sepal length = 6.6, sepal width = 3, petal lenhgth = 5.2, petal width = 2.3
new_iris = torch.tensor([6.6, 3, 5.2, 2.3])
#Now let our network evaluate it!
print("Evaluating New Data!")
with torch.no_grad():
    model(new_iris)
    name, name_index = name_output(torch)
    print(str(model(new_iris)) + "; Highest value: " + str(name_index) + "; It is a " + name) 

