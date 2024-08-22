# Vietnam stock price prediction
This project was developed to predict the future trends of stock prices in the Vietnamese stock market using deep learning. Specifically, I utilized an LSTM model, training it with the price data of 30 stocks from the VN30 index.
## LSTM model
The LSTM model will use the data of the previous 60 days to forecast the stock price at the next day. Accordingly, the LSTM model is built with a structure of 4 layers including the following specific coefficients:
- Layer 1: units = 30, activation = ‘relu’, Dropout(0.1), input shape corresponding to the specific data size of each stock code
- Layer 2: units = 40, activation = ‘relu’, Dropout(0.1)
- Layer 3: units = 50, activation = ‘relu’, Dropout(0.1)
- Layer 4: units = 60, activation = ‘relu’, Dropout(0.1)
## Dataset
I used [vnstock3](https://github.com/thinh-vu/vnstock) library to get Vietnam stock price. You can download with pip: 
```python
pip install vnstock3
```
The dataset is divided into two separate sets: training and testing. The training set includes data from the listing start date to January 02, 2013, the test set includes data from March 2, 2024 to May 31, 2024.
## Results
List of 7 comparison diagrams between real price and predicted price of 7 stocks in VN30 stock group:
![FPT stock](https://github.com/leducphuong19122004/Vietnam-Stock-Price-Prediction-/blob/master/diagram_image/FPT.png)
![GAS stock](https://github.com/leducphuong19122004/Vietnam-Stock-Price-Prediction-/blob/master/diagram_image/GAS.png)
![CTG stock](https://github.com/leducphuong19122004/Vietnam-Stock-Price-Prediction-/blob/master/diagram_image/CTG.png)


