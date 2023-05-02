# LSTM_anomaly_detection

Anomaly detection using LSTM.

# Daily Female Births Dataset

This dataset describes the number of daily female births in California in 1959. The units are a count and there are 365 observations. The source of the dataset is credited to Newton (1988).

Below is a plot of the entire dataset.

<img src="img/Daily-Female-Births-Dataset.png" width="700"/>

Dataset has two columns: Date and Births.

<img src="img/dataset_columns.png" width="400"/>

# Dataset Scaling and Split

Scaling dataset using MinMaxScaler.

<img src="img/data_scaling.png" width="800"/>

Splitting dataset into train and test sets.

<img src="img/dataset_split.png" width="800"/>

# Transform a time series into a prediction dataset

To do that we will use function create_dataset().

dataset: A numpy array of time series, first dimension is the time steps
lookback: Size of window for prediction

<img src="img/create_dataset.png" width="800"/>

# LSTM Model

Our model will have one LSTM layer with input_size = 1 and hidden_size = 50. And also one fully connected layer with 50 input channels and 1 output channel.

<img src="img/lstm.png" width="800"/>

# Training

We will train our model on 2000 epochs and use optimizer Adam and loss function MSE

<img src="img/training.png" width="800"/>

# Prediction

We will try to make a prediction and plot the results.

<img src="img/prediction.png" width="800"/>

# Anomaly Detection

To detect anomalies we need to calculate our model's errors and confidence intervals. If error is bigger than 1.5*(uncertanity) that is anomaly.

<img src="img/model_prediction.png" width="800"/>

In our case all observations are normal.

<img src="img/model_performance.png" width="800"/>
