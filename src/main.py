from generateInputs import InputsGenerator
from model import StockPricePredictionModel
from vnstock3 import Vnstock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    x_test, max_price, min_price = InputsGenerator("VNINDEX", start_date="2024-03-02", end_date="2024-05-31").generate_test_input()
    # x_train, y_train = InputsGenerator("VNINDEX").get_train_input()

    model = StockPricePredictionModel()
    lstm_model = model.setup_model()
    # model.train(lstm_model, x_train, y_train)

    lstm_model.load_weights('TrainedModel.weights.h5')
    price_prediction = lstm_model.predict(x_test)

    for i, price in enumerate(price_prediction):
        price_prediction[i][0] = price * (max_price - min_price) + min_price

    price_prediction = np.reshape(price_prediction, (-1))
    
    stock = Vnstock().stock(symbol="VNINDEX", source='VCI')
    df = stock.quote.history(start="2024-03-02", end="2024-05-31", interval='1D') # 61
    actual_price = df.open.values

    draw_diagram(predicted_price=price_prediction, actual_price=actual_price)
    

def draw_diagram(predicted_price, actual_price):
    plt.plot(actual_price, color='red', label='Actual stock price')
    plt.plot(predicted_price, color='blue', label='Predicted stock price')
    plt.title("Stock price prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()