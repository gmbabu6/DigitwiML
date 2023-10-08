import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class cElegans_DTwin:
    def __init__(self, earth_data, space_data):
        self.earth_data = earth_data
        self.space_data = space_data
        self.pretrain_model = self.create_model()
        self.finetune_model = self.create_model()
        
        # Split and normalize the data
        self.prepare_data()
        
    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def prepare_data(self):
        # Split data for pre-training and fine-tuning
        earth_pretrain, self.earth_finetune, space_pretrain, self.space_finetune = train_test_split(self.earth_data, 
                                                                                        self.space_data, test_size=0.1,
                                                                                                   random_state = 24)
        self.earth_train, self.earth_val, self.space_train, self.space_val = train_test_split(earth_pretrain, space_pretrain, 
                                                                                              test_size=0.11,
                                                                                                   random_state = 24)  # Roughly 10% of 90%

        # Normalize data
        self.scaler_earth = MinMaxScaler()
        self.scaler_space = MinMaxScaler()
        self.earth_train = self.scaler_earth.fit_transform(self.earth_train.reshape(-1, 1))
        self.earth_val = self.scaler_earth.transform(self.earth_val.reshape(-1, 1))
        self.earth_finetune = self.scaler_earth.transform(self.earth_finetune.reshape(-1, 1))
        self.space_train = self.scaler_space.fit_transform(self.space_train.reshape(-1, 1))
        self.space_val = self.scaler_space.transform(self.space_val.reshape(-1, 1))
        self.space_finetune = self.scaler_space.transform(self.space_finetune.reshape(-1, 1))
        
    def train(self):
        print("Pre-training model...")
        history_pretrain = self.pretrain_model.fit(self.earth_train, self.space_train, epochs=100, batch_size=16, validation_data=(self.earth_val, self.space_val), verbose=0)
        
        # Visualize pre-training
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history_pretrain.history['loss'], label='Train Loss')
        plt.plot(history_pretrain.history['val_loss'], label='Validation Loss')
        plt.title('Pre-training Losses')
        plt.legend()
        
        # Transfer learning for fine-tuning
        print("Fine-tuning model...")
        self.finetune_model.set_weights(self.pretrain_model.get_weights())
        history_finetune = self.finetune_model.fit(self.earth_finetune, self.space_finetune, epochs=50, batch_size=8, verbose=0)
        
        # Visualize fine-tuning
        plt.subplot(1, 2, 2)
        plt.plot(history_finetune.history['loss'], label='Fine-tuning Loss')
        plt.title('Fine-tuning Losses')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def validate(self):
        predictions = self.finetune_model.predict(self.earth_val)
        error = np.abs(predictions.reshape(-1) - self.space_val.reshape(-1)).mean()
        print(f"Mean Absolute Error on Validation Data: {error:.4f}")
        
        # Visualization for actual vs predictions
        plt.figure(figsize=(8, 6))
        plt.scatter(self.space_val, predictions, alpha=0.5)
        plt.plot([min(self.space_val), max(self.space_val)], [min(self.space_val), max(self.space_val)], 'r')
        plt.xlabel('Actual Space Gene Expressions')
        plt.ylabel('Predicted Gene Expressions')
        plt.title('Actual vs Predicted Gene Expressions')
        plt.grid(True)
        plt.show()
        
    def heuristic_fine_tuning(self):
        # Compute scaling factors from NN predictions on fine-tuning set
        predictions = self.finetune_model.predict(self.earth_finetune).reshape(-1)
        self.scaling_factors = self.space_finetune.reshape(-1) / predictions

        # Adjust predictions using scaling factors for validation set
        val_predictions = self.finetune_model.predict(self.earth_val).reshape(-1)
        adjusted_predictions = val_predictions * self.scaling_factors.mean() # Using mean scaling factor for simplicity
        
        # Visualization for actual vs adjusted predictions
        plt.figure(figsize=(8, 6))
        plt.scatter(self.space_val, adjusted_predictions, alpha=0.5)
        plt.plot([min(self.space_val), max(self.space_val)], [min(self.space_val), max(self.space_val)], 'r')
        plt.xlabel('Actual Space Gene Expressions')
        plt.ylabel('Adjusted Predicted Gene Expressions')
        plt.title('Actual vs Adjusted Predicted Gene Expressions (with Scaling Factors)')
        plt.grid(True)
        plt.show()
        
        error = np.abs(adjusted_predictions - self.space_val.reshape(-1)).mean()
        print(f"Mean Absolute Error after Second Fine-tuning: {error:.4f}")
        
    def analyze_new_data(self, new_earth_data, new_space_data):
        # Ensure new data is normalized in the same manner as training data
        normalized_new_data = self.scaler_earth.transform(new_earth_data.reshape(-1, 1))

        # Predict using the fine-tuned model
        predictions = self.finetune_model.predict(normalized_new_data).reshape(-1)

        # Adjust predictions using scaling factors
        adjusted_predictions = predictions * self.scaling_factors.mean() # Assuming scaling_factors is stored during fine-tuning

        # Visualization
        plt.figure(figsize=(15, 6))

        # Neural Network Predictions vs Actual
        plt.subplot(1, 2, 1)
        plt.scatter(new_space_data, predictions, alpha=0.5)
        plt.plot([min(new_space_data), max(new_space_data)], [min(new_space_data), max(new_space_data)], 'r')
        plt.xlabel('Actual Space Gene Expressions')
        plt.ylabel('Predicted Gene Expressions')
        plt.title('Actual vs NN Predicted Gene Expressions')
        plt.grid(True)

        # Adjusted Predictions vs Actual
        plt.subplot(1, 2, 2)
        plt.scatter(new_space_data, adjusted_predictions, alpha=0.5)
        plt.plot([min(new_space_data), max(new_space_data)], [min(new_space_data), max(new_space_data)], 'r')
        plt.xlabel('Actual Space Gene Expressions')
        plt.ylabel('Adjusted Predicted Gene Expressions')
        plt.title('Actual vs Adjusted Predicted Gene Expressions')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        return adjusted_predictions