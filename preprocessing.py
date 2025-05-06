import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
import json
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

class CASASDataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.output_dir = os.path.join(os.path.dirname(data_path), "processed")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define sensor layout for Aruba dataset
        self.sensor_mapping = {
            # Kitchen Area
            'M001': 'Kitchen_Stove', 'M002': 'Kitchen_Sink', 'M003': 'Kitchen_Fridge',
            'M004': 'Kitchen_Counter', 'M019': 'Kitchen_Table', 'M020': 'Kitchen_Table',
            # Living Room
            'M005': 'LivingRoom_Sofa', 'M006': 'LivingRoom_TV', 'M007': 'LivingRoom_Center',
            # Bedroom
            'M008': 'Bedroom_Dresser', 'M009': 'Bedroom_Bed', 'M010': 'Bedroom_Entrance',
            # Bathroom
            'M011': 'Bathroom_Sink', 'M012': 'Bathroom_Shower',
            # Doors
            'D001': 'FrontDoor', 'D002': 'BackDoor', 'D003': 'BedroomDoor',
            # Other
            'M013': 'Hallway', 'M014': 'DiningTable', 'M015': 'Pantry'
        }
        
        self.activity_mapping = {
            'Meal_Preparation': 'Cooking',
            'Wash_Dishes': 'Cleaning',
            'Eating': 'Dining',
            'Sleeping': 'Sleep',
            'Bed_to_Toilet': 'Sleep',
            'Enter_Home': 'Transition',
            'Leave_Home': 'Transition',
            'Relax': 'Leisure',
            'Work': 'Work',
            'Housekeeping': 'Cleaning'
        }

    def parse_raw_data(self):
        """Parse the raw CASAS data.txt file with flexible timestamp handling"""
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                # Handle different line formats
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                
                # Handle timestamp with and without milliseconds
                try:
                    timestamp = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    try:
                        timestamp = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M:%S")
                    except ValueError as e:
                        print(f"Skipping line due to timestamp error: {line.strip()}")
                        continue
                
                sensor = parts[2]
                value = parts[3]
                
                # Extract activity if present
                activity = None
                status = None
                if len(parts) >= 5:
                    if parts[4] in self.activity_mapping:
                        activity = parts[4]
                        if len(parts) >= 6:
                            status = parts[5]
                    elif parts[4] in ['begin', 'end']:
                        status = parts[4]
                
                data.append({
                    'timestamp': timestamp,
                    'sensor_id': sensor,
                    'value': value,
                    'activity': activity,
                    'status': status
                })
        
        return pd.DataFrame(data)

    def create_features(self, df):
        """Create temporal and spatial features"""
        # Temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        
        # Sensor features
        df['sensor_room'] = df['sensor_id'].map(self.sensor_mapping)
        df['sensor_type'] = df['sensor_id'].str[0].map({
            'M': 'motion', 'D': 'door', 'T': 'temperature'
        })
        
        return df

    def create_activity_segments(self, df, window_size='5T'):
        """Create time-based segments with dominant activity"""
        df = df.set_index('timestamp')
        
        # Resample to fixed windows
        activity_segments = df['activity'].resample(window_size).apply(
            lambda x: x.mode()[0] if not x.mode().empty else None)
        
        # Count sensor activations per window
        sensor_counts = df.groupby([pd.Grouper(freq=window_size), 'sensor_id']).size().unstack(fill_value=0)
        
        # Combine features
        features = pd.DataFrame({
            'dominant_activity': activity_segments,
            'activity_changes': df['activity'].resample(window_size).nunique()
        }).join(sensor_counts, how='left').fillna(0)
        
        # Forward fill activities
        features['dominant_activity'] = features['dominant_activity'].ffill()
        
        return features.reset_index()

    def build_sensor_graph(self):
        """Create a graph representation of sensor relationships"""
        G = nx.Graph()
        
        # Add nodes with attributes
        for sensor, location in self.sensor_mapping.items():
            room, area = location.split('_') if '_' in location else (location, '')
            G.add_node(sensor, room=room, area=area, type=sensor[0])
        
        # Add edges based on physical proximity (simplified)
        room_groups = {}
        for node in G.nodes():
            room = G.nodes[node]['room']
            if room not in room_groups:
                room_groups[room] = []
            room_groups[room].append(node)
        
        for room, sensors in room_groups.items():
            for i in range(len(sensors)):
                for j in range(i+1, len(sensors)):
                    G.add_edge(sensors[i], sensors[j], weight=1.0)
        
        return G

    def process_pipeline(self):
        """Run full processing pipeline"""
        print(f"Processing data from: {self.data_path}")
        
        # Step 1: Parse raw data
        raw_df = self.parse_raw_data()
        print(f"Parsed {len(raw_df)} raw events")
        
        # Step 2: Create features
        feature_df = self.create_features(raw_df)
        
        # Step 3: Create time segments
        window_df = self.create_activity_segments(feature_df)
        print(f"Created {len(window_df)} time windows")
        
        # Step 4: Build sensor graph
        sensor_graph = self.build_sensor_graph()
        print(f"Built sensor graph with {len(sensor_graph.nodes())} nodes")
        
        # Step 5: Ensure consistent data types before saving
        window_df['dominant_activity'] = window_df['dominant_activity'].astype(str)
        
        # Step 6: Save processed data
        window_df.to_parquet(os.path.join(self.output_dir, "window_features.parquet"))
        nx.write_graphml(sensor_graph, os.path.join(self.output_dir, "sensor_graph.graphml"))
        
        with open(os.path.join(self.output_dir, "sensor_mapping.json"), 'w') as f:
            json.dump(self.sensor_mapping, f)
        
        print(f"Processing complete. Results saved to: {self.output_dir}")
        return window_df, sensor_graph

if __name__ == "__main__":
    processor = CASASDataProcessor(r"C:\Users\User\Downloads\aruba\data.txt")
    features, graph = processor.process_pipeline()
    