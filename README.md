### Developed by: Naveen Kumar S
### Register Number: 212221240033
### Date:

# Ex.No: 1B CONVERSION OF NON STATIONARY TO STATIONARY DATA

### AIM:
To perform regular differncing,seasonal adjustment and log transformation on Stock Price data of NVIDIA.

### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.

### PROGRAM:
```py
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Load the dataset
nvidia_df = pd.read_csv('C:/Users/lenovo/Downloads/archive (2)/NVIDIA/NvidiaStockPrice.csv')

# Convert the 'Date' column to datetime format
nvidia_df['Date'] = pd.to_datetime(nvidia_df['Date'])

# Set the 'Date' column as index
nvidia_df.set_index('Date', inplace=True)

# Select 'Adj Close' for analysis
nvidia_series = nvidia_df['Adj Close'].dropna()
```

#### REGULAR DIFFERENCING:

```python
# Differencing to remove trend (regular differencing)
nvidia_diff = nvidia_series.diff().dropna()

# Plot the differenced series
plt.figure(figsize=(10, 6))
plt.plot(nvidia_diff, label='Differenced Series', color='green')
plt.title('NVIDIA Stock Price (Differenced)')
plt.xlabel('Date')
plt.ylabel('Differenced Adj Close Price')
plt.legend()
plt.grid(True)
plt.show()
```
#### SEASONAL ADJUSTMENT:

```python
# Seasonal decomposition (additive model)
decomposition = seasonal_decompose(nvidia_log, model='additive', period=252)  # Assuming 252 trading days per year

# Plot the decomposed components
decomposition.plot()
plt.suptitle('Seasonal Decomposition of NVIDIA Log Transformed Series', fontsize=16)
plt.show()
```
#### LOG TRANSFORMATION:

```python
# Perform Log Transformation to stabilize variance
nvidia_log = np.log(nvidia_series)

# Plot the log-transformed series
plt.figure(figsize=(10, 6))
plt.plot(nvidia_log, label='Log Transformed Series', color='orange')
plt.title('NVIDIA Stock Price (Log Transformed)')
plt.xlabel('Date')
plt.ylabel('Log(Adj Close Price)')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT:

#### ORIGINAL PLOT:
![image](https://github.com/user-attachments/assets/2a437ade-bc24-43ff-b72a-4523a89b0e47)

#### REGULAR DIFFERENCING:
![image](https://github.com/user-attachments/assets/bae01fb3-2955-40ab-b297-330bc78f6d3b)

#### SEASONAL ADJUSTMENT:

![image](https://github.com/user-attachments/assets/5d84ab7f-0466-47dc-9e3d-88ae1daa3ea2)

#### LOG TRANSFORMATION:
![image](https://github.com/user-attachments/assets/c550291e-e013-450d-baba-93b94125e146)


### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on Stock Price data of NVIDIA.
