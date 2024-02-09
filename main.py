import numpy as np # linear algebra
import pandas as pd # for data processing, CSV file I/O 
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

data = pd.read_csv("/content/TSLA.csv")

# To check whether data is fitted succesfully or not
print(data.head())


#information of the whole dataset
print(data.info())

## Spliting Data as Train and Validation

length_data = len(data)     # rows that data has
split_ratio = 0.7           # 70%  train + 30% validation
length_train = round(length_data * split_ratio)  
length_validation = length_data - length_train
print("Data length :", length_data)
print("Train data length :", length_train)
print("Validation data lenth :", length_validation)

# Preparing training data
train_data = data[:length_train].iloc[:,:2] 
train_data['Date'] = pd.to_datetime(train_data['Date'])  # converting to date time object
print("training data :",train_data)

# Preparing validation data
validation_data = data[length_train:].iloc[:,:2]
validation_data['Date'] = pd.to_datetime(validation_data['Date'])  
print("validation data :",validation_data)

###  Creating Train Dataset from Train split

# We will get Open column as our dataset
# Dataset to be converted to array by adding .values

dataset_train = train_data.Open.values
print("Before changing Dimension of Dataset train shape :",dataset_train.shape)

# Change 1d array to 2d array
# Changing shape from (1692,) to (1692,1)

dataset_train = np.reshape(dataset_train, (-1,1))
print("After changing Dimension of Dataset train shape :",dataset_train.shape)

### Normalization / Feature Scaling

#Dataset values will be in between 0 and 1 after scaling

scaler = MinMaxScaler(feature_range = (0,1))

# scaling dataset
dataset_train_scaled = scaler.fit_transform(dataset_train)
print("Scaled Dataset shape:",dataset_train_scaled.shape)

## Plotting the dataset using matplotlib
plt.subplots(figsize = (15,6))
plt.plot(dataset_train_scaled)
plt.xlabel("Days as 1st, 2nd, 3rd..")
plt.ylabel("Open Price")
plt.show()


### Creating X_train and y_train from Train data

# We have train data composed of stock open prices over days
# So, it has 1184 prices corresponding 1184 days
# My aim is to predict the open price of the next day.
# I can use a time step of 50 days.
# I will pick first 50 open prices (0 to 50), 1st 50 price will be in X_train data
# Then predict the price of 51th day; and 51th price will be in y_train data
# Again, i will pick prices from 1 to 51, those will be in X_train data
# Then predict the next days price, 52nd price will be in y_train data

x_train = []
y_train = []

time_step = 50

for i in range(time_step, length_train):
    x_train.append(dataset_train_scaled[i-time_step:i,0])
    y_train.append(dataset_train_scaled[i,0])
    
# convert list to array
x_train, y_train = np.array(x_train), np.array(y_train)

print("Shape of X_train before reshape :",x_train.shape)
print("Shape of y_train before reshape :",y_train.shape)

### Reshape

X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
y_train = np.reshape(y_train, (y_train.shape[0],1))

print("Shape of X_train after reshape :",x_train.shape)
print("Shape of y_train after reshape :",y_train.shape)

# Shape of X_train : 1134 x 50 x 1
# That means we have 1134 rows, each row has 50 rows and 1 column
# Lets check the first row: it has 50 rows (open prices of 49 days)

print(x_train[0])
print(y_train[0])

### Creating RNN model

# importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

# initializing the RNN
regressor = Sequential()

# adding first RNN layer and dropout regulatization
regressor.add(
    SimpleRNN(units = 50, 
              activation = "tanh", 
              return_sequences = True, 
              input_shape = (X_train.shape[1],1))
             )

regressor.add(
    Dropout(0.2))
# adding second RNN layer and dropout regulatization

regressor.add(
    SimpleRNN(units = 50, 
              activation = "tanh", 
              return_sequences = True))
regressor.add(
    Dropout(0.2))

# adding third RNN layer and dropout regulatization

regressor.add(
    SimpleRNN(units = 50, 
              activation = "tanh", 
              return_sequences = True))

regressor.add(
    Dropout(0.2))

# adding fourth RNN layer and dropout regulatization

regressor.add(
    SimpleRNN(units = 50))

regressor.add(
    Dropout(0.2))

# adding the output layer
regressor.add(Dense(units = 1))

# compiling RNN
regressor.compile(
    optimizer = "adam", 
    loss = "mean_squared_error",
    metrics = ["accuracy"])

# fitting the RNN
history = regressor.fit(x_train, y_train, epochs = 20, batch_size = 32)
# print(history)

#### Evaluating Model

# Losses
print("Losses :",history.history["loss"])

# Plotting Loss vs Epochs
plt.figure(figsize =(10,7))
plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("Simple RNN model, Loss vs Epoch")
plt.show()

# Plotting Accuracy vs Epochs
plt.figure(figsize =(10,5))
plt.plot(history.history["accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracies")
plt.title("Simple RNN model, Accuracy vs Epoch")
plt.show()

### Model predictions for train data

y_pred = regressor.predict(X_train)  # predictions
y_pred = scaler.inverse_transform(y_pred) # scaling back from 0-1 to original
print("shape of predicted data :", y_pred.shape)

y_train = scaler.inverse_transform(y_train) 
print("y training shape :",y_train.shape)

###  Visualisation
plt.figure(figsize = (30,10))
plt.plot(y_pred, color = "b", label = "y_pred" )
plt.plot(y_train, color = "g", label = "y_train")
plt.xlabel("Days")
plt.ylabel("Open price")
plt.title("Simple RNN model, Predictions with input X_train vs y_train")
plt.legend()
plt.show()

### Creating Test Dataset from Validation Data

# Converting array and scaling
dataset_validation = validation_data.Open.values  # getting "open" column and converting to array
dataset_validation = np.reshape(dataset_validation, (-1,1))  # converting 1D to 2D array
scaled_dataset_validation =  scaler.fit_transform(dataset_validation)  # scaling open values to between 0 and 1
print("Shape of scaled validation dataset :",scaled_dataset_validation.shape)

# Creating x_test and y_test

x_test = []
y_test = []

for i in range(time_step, length_validation):
    x_test.append(scaled_dataset_validation[i-time_step:i,0])
    y_test.append(scaled_dataset_validation[i,0])

# Converting to array
x_test, y_test = np.array(x_test), np.array(y_test)

print("Shape of X_test before reshape :",x_test.shape)
print("Shape of y_test before reshape :",y_test.shape)

# Reshape

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))  # reshape to 3D array
y_test = np.reshape(y_test, (-1,1))  # reshape to 2D array
print("Shape of X_test after reshape :",x_test.shape)
print("Shape of y_test after reshape :",y_test.shape)

### Evaluating with Validation Data

# predictions with X_test data
y_pred_of_test = regressor.predict(x_test)
# scaling back from 0-1 to original
y_pred_of_test = scaler.inverse_transform(y_pred_of_test) 
print("Shape of y_pred_of_test :",y_pred_of_test.shape)

### Visualisation
plt.figure(figsize = (30,10))
plt.plot(y_pred_of_test, label = "y_pred_of_test", c = "orange")
plt.plot(scaler.inverse_transform(y_test), label = "y_test", c = "g")
plt.xlabel("Days")
plt.ylabel("Open price")
plt.title("Simple RNN model, Prediction with input X_test vs y_test")
plt.legend()
plt.show()

### Visualisation
plt.subplots(figsize =(30,12))
plt.plot(train_data.Date, train_data.Open, label = "train_data", color = "b")
plt.plot(validation_data.Date, validation_data.Open, label = "validation_data", color = "g")
plt.plot(train_data.Date.iloc[time_step:], y_pred, label = "y_pred", color = "r")
plt.plot(validation_data.Date.iloc[time_step:], y_pred_of_test, label = "y_pred_of_test", color = "orange")
plt.xlabel("Days")
plt.ylabel("Open price")
plt.title("Simple RNN model, Train-Validation-Prediction")
plt.legend()
plt.show()

### CREATING A "LSTM" MODEL

y_train = scaler.fit_transform(y_train)
from keras.layers import LSTM

model_lstm = Sequential()
model_lstm.add(
    LSTM(64,return_sequences=True,input_shape = (X_train.shape[1],1))) #64 lstm neuron block
model_lstm.add(
    LSTM(64, return_sequences= False))
model_lstm.add(Dense(32))
model_lstm.add(Dense(1))
model_lstm.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])
history2 = model_lstm.fit(X_train, y_train, epochs = 10, batch_size = 10)

### Evaluating LSTM Model

plt.figure(figsize =(10,5))
plt.plot(history2.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("LSTM model, Accuracy vs Epoch")
plt.show()

plt.subplots(figsize =(30,12))
plt.plot(scaler.inverse_transform(model_lstm.predict(X_test)), label = "y_pred_of_test", c = "orange" )
plt.plot(scaler.inverse_transform(y_test), label = "y_test", color = "g")
plt.xlabel("Days")
plt.ylabel("Open price")
plt.title("LSTM model, Predictions with input X_test vs y_test")
plt.legend()
plt.show()

### Future price prediction

# Which day is the last day in our data?
print(data.iloc[-1])
# We can predict the open price for the day after 2022-03-24--> for 2023-03-24.
x_input = data.iloc[-time_step:].Open.values               # getting last 50 rows and converting to array
x_input = scaler.fit_transform(x_input.reshape(-1,1))      # converting to 2D array and scaling
x_input = np.reshape(x_input, (1,50,1))                    # reshaping : converting to 3D array
print("Shape of X_input :", x_input.shape)
print("x_input data :",x_input)

simple_RNN_prediction = scaler.inverse_transform(regressor.predict(x_input))
LSTM_prediction = scaler.inverse_transform(model_lstm.predict(x_input))
print("Simple RNN, Open price prediction for 2022-03-24      :", simple_RNN_prediction[0,0])
print("LSTM prediction, Open price prediction for 2022-03-24 :", LSTM_prediction[0,0])