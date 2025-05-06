import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, MultiHeadAttention, 
                                    LayerNormalization, Dropout, Concatenate)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

class HARModeler:
    def __init__(self, features_path, graph_path):
        self.features = pd.read_parquet(features_path)
        self.graph = nx.read_graphml(graph_path)
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data for modeling with proper label handling"""
        # Filter out unknown activities
        self.features = self.features[self.features['dominant_activity'].notna()]
        
        # Get all unique activities that exist in the data
        self.unique_activities = sorted(self.features['dominant_activity'].unique())
        
        # Create a label encoder that knows all possible activities
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.unique_activities)
        
        # Encode activities numerically
        encoded_labels = self.label_encoder.transform(self.features['dominant_activity'])
        
        # Convert to one-hot encoding
        self.activity_labels = pd.get_dummies(pd.Series(encoded_labels)).values
        self.num_classes = len(self.unique_activities)
        
        # Prepare sensor features
        sensor_cols = [col for col in self.features.columns 
                      if col.startswith('M') or col.startswith('D')]
        self.sensor_features = self.features[sensor_cols]
        
        # Normalize
        self.scaler = MinMaxScaler()
        self.sensor_features = pd.DataFrame(
            self.scaler.fit_transform(self.sensor_features),
            columns=self.sensor_features.columns
        )

    class TransformerBlock(tf.keras.layers.Layer):
        """Transformer encoder block with proper dimension handling"""
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.rate = rate
            
            # Initialize layers in build() instead of __init__
            self.att = None
            self.ffn = None
            self.layernorm1 = None
            self.layernorm2 = None
            self.dropout1 = None
            self.dropout2 = None

        def build(self, input_shape):
            # Initialize layers with proper shapes
            self.att = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
            self.ffn = tf.keras.Sequential([
                Dense(self.ff_dim, activation="relu"),
                Dense(input_shape[-1])  # Match input dimension
            ])
            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)
            self.dropout1 = Dropout(self.rate)
            self.dropout2 = Dropout(self.rate)
            super().build(input_shape)

        def call(self, inputs, training=False):
            # Attention block
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            
            # Feed forward block
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            
            return self.layernorm2(out1 + ffn_output)

        def compute_output_shape(self, input_shape):
            return input_shape

    def build_hybrid_model(self):
        """Build the hybrid Transformer-LSTM model with proper dimensions"""
        # Sequence input
        seq_input = Input(shape=(None, self.sensor_features.shape[1]))
        
        # Transformer branch
        transformer_block = self.TransformerBlock(
            embed_dim=self.sensor_features.shape[1],  # Match input dimension
            num_heads=4,
            ff_dim=128
        )
        x = transformer_block(seq_input)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # LSTM branch
        y = LSTM(128, return_sequences=True)(seq_input)
        y = LSTM(64)(y)
        
        # Fusion
        combined = Concatenate()([x, y])
        outputs = Dense(self.num_classes, activation="softmax")(combined)
        
        model = Model(inputs=seq_input, outputs=outputs)
        model.compile(
            optimizer=Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def create_sequences(self, window_size=10):
        """Create input sequences for the model"""
        X, y = [], []
        for i in range(len(self.sensor_features) - window_size):
            X.append(self.sensor_features.iloc[i:i+window_size].values)
            y.append(self.activity_labels[i+window_size])
        return np.array(X), np.array(y)

    def train_evaluate(self):
        """Train and evaluate the model with proper label handling"""
        X, y = self.create_sequences()
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        histories = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"\nTraining Fold {fold + 1}")
            
            model = self.build_hybrid_model()
            history = model.fit(
                X[train_idx],
                y[train_idx],
                validation_data=(X[test_idx], y[test_idx]),
                epochs=15,
                batch_size=32,
                verbose=1
            )
            histories.append(history)
            
            # Evaluate
            y_pred = model.predict(X[test_idx])
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y[test_idx], axis=1)
            
            print(f"\nFold {fold + 1} Classification Report:")
            print(classification_report(
                y_true,
                y_pred_classes,
                target_names=self.label_encoder.classes_,
                labels=np.arange(len(self.label_encoder.classes_))
            ))
        
        return model, histories

    def save_model(self, model, output_dir):
        """Save the trained model"""
        os.makedirs(output_dir, exist_ok=True)
        model.save(os.path.join(output_dir, "har_model.h5"))
        
        # Save the label encoder for future use
        import joblib
        joblib.dump(self.label_encoder, os.path.join(output_dir, "label_encoder.joblib"))
        
        print(f"Model saved to {os.path.join(output_dir, 'har_model.h5')}")
        print(f"Label encoder saved to {os.path.join(output_dir, 'label_encoder.joblib')}")

if __name__ == "__main__":
    data_dir = r"C:\Users\User\Downloads\aruba\processed"
    modeler = HARModeler(
        os.path.join(data_dir, "window_features.parquet"),
        os.path.join(data_dir, "sensor_graph.graphml")
    )
    model, histories = modeler.train_evaluate()
    modeler.save_model(model, os.path.join(data_dir, "models"))