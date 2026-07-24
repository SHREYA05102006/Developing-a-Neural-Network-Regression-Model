# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression is a supervised machine learning task used to predict continuous numerical values, such as house prices, salaries, temperatures, or sales.

Traditional regression methods like Linear Regression work well only when the relationship between the input and output is linear. However, many real-world datasets have complex and non-linear relationships between variables. In such cases, Neural Networks provide better prediction accuracy because they can learn these complex patterns from the data.

In this experiment, a Neural Network Regression Model is developed using a dataset containing input features and a continuous target variable. The dataset is divided into training and testing sets. The neural network learns by adjusting its weights during training using backpropagation and an optimizer such as Adam. After training, the model predicts the target values for unseen test data, and its performance is evaluated using regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

The objective of this experiment is to build an accurate regression model capable of predicting continuous values from the given input features.

## Neural Network Model
<img width="495" height="406" alt="image" src="https://github.com/user-attachments/assets/a36e647d-0f95-42cd-9071-834122536f7e" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:V.SHREYA

### Register Number:212224230266

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\Akbar\Downloads\deep learning exp\Exp-1.csv")
df

x = df[["Input"]].values
y = df[["Output"]].values
xt,xst,yt,yst = train_test_split(x,y,test_size=0.2,random_state=42)

scale1 = MinMaxScaler()
scale2=MinMaxScaler()
xt = scale1.fit_transform(xt)
xst = scale2.fit_transform(xst)


xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)
yt = torch.FloatTensor(yt)
yst = torch.FloatTensor(yst)

class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(), 
            nn.Linear(16,8), 
            nn.ReLU(), 
            nn.Linear(8,1)
        )
    def forward(self,x):
        return self.network(x)

# Initialize the Model, Loss Function, and Optimizer
model = neuralnet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# Train the model
epochs = 1000
losses=[]
for i in range(epochs):
    optimizer.zero_grad()
    pred = model(xt)
    loss = criterion(pred, yt)
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"{i}/{epochs} Loss: {loss.item():.4f}")
        losses.append(loss.item())

# Tesing for new input
new = scale1.transform([[16]])
new = torch.FloatTensor(new)

pred = model(new)
print(pred.item())

# Evaluating loss for testing data
with torch.no_grad():
    pred=model(xst)
    loss_test=criterion(pred,yst)
    print(loss_test)

# Plot the loss curve

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


```

### Dataset Information
<img width="851" height="652" alt="image" src="https://github.com/user-attachments/assets/63e71274-0e64-4a9b-a58f-2a3f7a37891d" />


### OUTPUT

### Training Loss Vs Iteration Plot
<img width="872" height="653" alt="image" src="https://github.com/user-attachments/assets/5f50e649-5995-478b-8c5f-260c155b63c1" />


### New Sample Data Prediction
<img width="593" height="707" alt="image" src="https://github.com/user-attachments/assets/d3c48fa2-f1d2-4e03-aba4-a3949a092ad8" />

<img width="603" height="427" alt="image" src="https://github.com/user-attachments/assets/fb7a8237-89ea-4810-a0ac-c3c23d728d6e" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
