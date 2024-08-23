### Developed by: Syed Abdul Wasih H
### Register Number: 212221240057
### Date:

# Ex.No: 1B CONVERSION OF NON STATIONARY TO STATIONARY DATA
### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.


### PROGRAM:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('international-airline-passengers.csv')
x=data.iloc[:,0]
y=data.iloc[:,1]
plt.xlabel('Month')
plt.ylabel('No. of passengers in thousands.')
plt.plot(x,y)

```

#### REGULAR DIFFERENCING:

```python
data1=data
data1['diff']=data1.iloc[:,1].diff(periods=1)
data1=data1.dropna()
x=data1['Month']
y=data1['diff']
plt.xlabel('Month')
plt.ylabel('No. of passengers in thousands.')
plt.plot(x,y)

```

#### SEASONAL ADJUSTMENT:

```python
data2=data
data2['SeasonalAdjustment'] = data2.iloc[:,1] - data2.iloc[:,1].shift(12)
data2['SeasonalAdjustment'].dropna()
x=data2['Month']
y=data2["SeasonalAdjustment"]
plt.xlabel('Month')
plt.ylabel('No. of passengers in thousands.')
plt.plot(x,y)

```

#### LOG TRANSFORMATION:

```python
data3=data
data3['log']=np.log(data3['diff']).dropna()
data3=data3.dropna()
x=data3['Month']
y=data3['log']
plt.xlabel('Month')
plt.ylabel('No. of passengers in thousands.')
plt.plot(x,y)
```

### OUTPUT:

#### REGULAR DIFFERENCING:

![tsa1](https://github.com/anto-richard/TSA_EXP1B/assets/93427534/7d84d964-afaf-4afa-b575-8c2dcd6d27d1)

#### SEASONAL ADJUSTMENT:

![tsa2](https://github.com/anto-richard/TSA_EXP1B/assets/93427534/86478229-d90e-4251-bf2e-82c9a222b3a0)

#### LOG TRANSFORMATION:

![tsa3](https://github.com/anto-richard/TSA_EXP1B/assets/93427534/f3b1a060-d2a9-44ee-9b4f-cb599ba5c546)
![tsa4](https://github.com/anto-richard/TSA_EXP1B/assets/93427534/b40b39fe-5a63-48ae-bdf8-f68a7bf2f104)


### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
