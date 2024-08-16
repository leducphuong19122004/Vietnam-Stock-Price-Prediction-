from vnstock3 import Vnstock
import numpy as np
import pandas as pd

class InputsGenerator:
    def __init__(self, stock_code, start_date='2013-01-02', end_date='2024-03-01') -> None:
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date

    def get_train_input(self):
        stock = Vnstock().stock(symbol=self.stock_code, source='VCI')
        df = stock.quote.history(start=self.start_date, end=self.end_date, interval='1D')

        min_price = df.open[0]
        max_price = df.open[0]
        vnindex_price_list = []

        for i, price in enumerate(df.open):
            if price > max_price:
                max_price = price
            if price < min_price:
                min_price = price
            vnindex_price_list.append(price)

        # normalize data set, set range (0,1)
        vnindex_dataset = np.ones((2782, 1))
        for i, price in enumerate(vnindex_price_list):
            vnindex_dataset[i] = (price - min_price) / (max_price - min_price)

        # the LSTM model will use the data of the previous 60 days
        # to forecast the stock price at the next day.
        x_train = []
        y_train = []
        for i in range(60, 2782):
            x_train.append(vnindex_dataset[i-60:i, 0])
            y_train.append(vnindex_dataset[i, 0])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # reshape x_train to match input lstm model
        x_train = np.expand_dims(x_train, axis=-1)
        # print(x_train.shape) #(2722, 60, 1)
        # print(y_train.shape) #(2722,)
        return (x_train, y_train)
    
    def generate_test_input(self):
        stock = Vnstock().stock(symbol=self.stock_code, source='VCI')

        test_dataset = stock.quote.history(start=self.start_date, end=self.end_date, interval='1D') # 61
        train_dataset = stock.quote.history(start="2013-01-02", end="2024-03-01", interval='1D') # 2782
        dataset_total = pd.concat((train_dataset.open, test_dataset.open), axis=0) # length = 2843
        inputs = dataset_total[len(dataset_total) - len(test_dataset) - 60:].values
        
        # normalize and reshape inputs
        min_input = inputs[0]
        max_input = inputs[0]
        for price in inputs:
            if price < min_input:
                min_input = price
            if price > max_input:
                max_input = price
        reshape_inputs = np.ones((len(inputs), 1))
        for i, price in enumerate(inputs):
            reshape_inputs[i] = (price - min_input) / (max_input - min_input)
        
        # create input `x_test` for model.predict() method
        x_test = []
        for i in range(60, len(inputs)):
                x_test.append(reshape_inputs[i-60:i, 0])
                
        x_test = np.array(x_test)
        x_test = np.expand_dims(x_test, axis=-1)
        
        return x_test, max_input, min_input