import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class LSTMForecaster:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model
    
    def fit(self, data, epochs=50, batch_size=32, validation_split=0.1):
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Build and train the model
        if self.model is None:
            self.build_model((self.sequence_length, 1))
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        return history
    
    def predict(self, data):
        # Scale the data
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        # Create sequences
        X, _ = self.create_sequences(scaled_data)
        
        # Make predictions
        scaled_predictions = self.model.predict(X)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(scaled_predictions)
        return predictions
    
    def forecast(self, data, steps):
        # Scale the data
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        # Initialize predictions array
        predictions = []
        current_sequence = scaled_data[-self.sequence_length:]
        
        for _ in range(steps):
            # Reshape sequence for prediction
            current_sequence_reshaped = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Make prediction
            next_pred = self.model.predict(current_sequence_reshaped, verbose=0)
            
            # Inverse transform prediction
            next_pred_original = self.scaler.inverse_transform(next_pred)
            predictions.append(next_pred_original[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
        
        return np.array(predictions) 