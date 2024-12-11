import torch
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()

df = pd.DataFrame(data=wine['data'], columns=wine['feature_names'])
pd.set_option('display.max_columns', None)
print(df)

features =  13# use 13 features

X_train, X_test, y_train, y_test = train_test_split(
    wine.data[:, :features], 
    wine.target, 
    test_size=0.3, 
    shuffle=True,
    random_state=42 
)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

class WineNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden_neurons):
        super(WineNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_input, n_hidden_neurons)
        self.activ1 = torch.nn.ReLU() 
        self.fc2 = torch.nn.Linear(n_hidden_neurons, int(n_hidden_neurons / 2)) 
        self.activ2 = torch.nn.ReLU()  
        self.fc3 = torch.nn.Linear(int(n_hidden_neurons / 2), 3)  
        self.sm = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activ1(x)
        x = self.fc2(x)
        x = self.activ2(x)
        x = self.fc3(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x
    
n_input =  13 # choose number of input neurons
n_hidden =  169 # choose number of hidden neurons
wine_net = WineNet(n_input, n_hidden)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(wine_net.parameters(), lr=1.0e-3)

batch_size = 32 # choose different batch sizes

for epoch in range(2000):
    order = np.random.permutation(len(X_train))
    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        
        batch_indexes = order[start_index:start_index+batch_size]
        
        x_batch = X_train[batch_indexes]
        y_batch = y_train[batch_indexes]
        
        preds = wine_net.forward(x_batch) 
        
        loss_value = loss(preds, y_batch)
        loss_value.backward()
        
        optimizer.step()
        
    if epoch % 10 == 0:
        test_preds = wine_net.forward(X_test)
        test_preds = test_preds.argmax(dim=1)

print(wine_net.fc1.in_features, np.asarray((test_preds == y_test).float().mean()) > 0.8)
# need to get 13 True
