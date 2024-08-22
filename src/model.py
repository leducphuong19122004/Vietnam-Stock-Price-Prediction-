from keras.api.layers import LSTM, Dense, Dropout, Input
from keras.api.models import Sequential, Model
from keras.api.callbacks import ModelCheckpoint, EarlyStopping
from os import mkdir

class StockPricePredictionModel:
    def __init__(self) -> None:
        pass

    def setup_model(self) -> Model:
        model = Sequential()
        model.add(Input(shape=(60, 1), dtype='float32')) # input_shape = (timesteps, features)
        # layer 1
        model.add(LSTM(30, return_sequences=True, activation='relu'))
        model.add(Dropout(0.1))
        # layer 2
        model.add(LSTM(40, return_sequences=True, activation='relu'))
        model.add(Dropout(0.1))
        # layer 3
        model.add(LSTM(50, return_sequences=True, activation='relu'))
        model.add(Dropout(0.1))
        # layer 4
        model.add(LSTM(60, activation='relu'))
        model.add(Dropout(0.1))

        model.add(Dense(units=1))
        return model

    def train(self, model: Model, x_train, y_train,stock_code: str, epochs=100, batch_size=32):
        print(f"[TRAINING] stock code: {stock_code}")
        model.compile(optimizer='adam',loss='mean_squared_error')
        
        file_path = f'trained_model/{stock_code}_model.weights.h5'
        model_checkpoint_callback = ModelCheckpoint(filepath=file_path, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(monitor='loss', patience=8)

        model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, shuffle=True, callbacks=[model_checkpoint_callback, early_stopping])

