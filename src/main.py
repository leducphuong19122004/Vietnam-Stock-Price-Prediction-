from generate_input import InputsGenerator
from model import StockPricePredictionModel
from vnstock3 import Vnstock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import mkdir, getcwd, listdir
import pathlib

def main():
    model = StockPricePredictionModel()
    lstm_model = model.setup_model()

    """ Training model for 30 stocks"""
    mkdir("trained_model")

    stock = Vnstock().stock("VN30", source="VCI")
    for stock_code in stock.listing.symbols_by_group("VN30"):
        x_train, y_train = InputsGenerator(stock_code=stock_code).get_train_input()

        model.train(lstm_model, x_train, y_train, stock_code=stock_code)
    
    """ Make prediction """
    for model_filepath in listdir("trained_model"):
        stock_code = model_filepath.split("_")[0]
        # get predicted stock price
        x_test, max_price, min_price = InputsGenerator(stock_code, start_date="2024-03-02", end_date="2024-05-31").generate_test_input()
        lstm_model.load_weights(f"trained_model/{model_filepath}")
    
        price_prediction = lstm_model.predict(x_test)
        for i, price in enumerate(price_prediction):
            price_prediction[i][0] = price * (max_price - min_price) + min_price

        price_prediction = np.reshape(price_prediction, (-1))
        
        # get real stock price
        stock = Vnstock().stock(symbol=stock_code, source='VCI')
        df = stock.quote.history(start="2024-03-02", end="2024-05-31", interval='1D') # 61
        real_price = df.open.values
        # draw a diagram to compare predicted price and real price
        draw_diagram(price_prediction, real_price, stock_code)
    

def draw_diagram(predicted_price: np.ndarray, real_price: np.ndarray, stock_code: str):
    plt.plot(real_price, color='red', label='Actual stock price')
    plt.plot(predicted_price, color='blue', label='Predicted stock price')
    plt.title(f"{stock_code} stock price prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()