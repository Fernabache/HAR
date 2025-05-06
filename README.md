# Human Activity Recognition for Smart Homes

A robust, generalizable approach to recognize activities across different smart home environments through functional abstraction and ensemble modeling.

## Overview

This project implements a Human Activity Recognition (HAR) system that can effectively recognize Activities of Daily Living (ADLs) across different smart home environments without requiring extensive per-home training. By applying functional abstraction to sensor data and employing an ensemble of models, the system achieves 93.6% accuracy while requiring 95% less training data than traditional approaches.

## Features

- **Sensor Abstraction Engine**: Transforms environment-specific data into functional representations
- **Cyclical Time Encoding**: Captures time-of-day patterns without creating artificial boundaries
- **Multi-model Ensemble**: Combines predictions from NBC, HMM, CRF, and LSTM models
- **Cross-Environment Generalization**: Works across different home layouts and sensor configurations
- **Privacy Preservation**: Edge computing implementation that protects user privacy

## Dataset

This project uses the CASAS Dataset 17, which contains:
- 1.6 million sensor events from 32 sensors
- 2 months of activity data
- 11 labeled activities including Meal_Preparation, Sleeping, and Bed_to_Toilet

## Architecture

### Data Processing Pipeline

```
Raw Data → Preprocessing → Feature Engineering → Sensor Abstraction → Model Training → Ensemble Prediction
```

### Key Components

1. **SensorAbstraction Class**: Maps specific sensor IDs to functional zones
2. **ArubaDatasetProcessor**: Handles data loading, cleaning, and feature extraction
3. **Model Implementations**:
   - Naive Bayes Classifier (82.1% accuracy)
   - Hidden Markov Model (for sequential pattern detection)
   - Conditional Random Field (93.3% accuracy)
   - LSTM Neural Network (90.7% accuracy)
4. **Ensemble Model**: Confidence-weighted voting mechanism

## Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| NBC   | 82.10%   | 79.40%    | 77.20% | 78.30%   |
| HMM   | 15.20%*  | 14.30%    | 13.70% | 14.00%   |
| CRF   | 93.30%   | 91.50%    | 90.80% | 91.10%   |
| LSTM  | 90.70%   | 89.20%    | 90.10% | 89.60%   |
| Ensemble | 93.60% | 92.20%   | 92.10% | 92.10%   |

*HMM is used specifically for sequential pattern detection rather than as a standalone classifier

### Cross-Home Generalization

| Layout Transfer | Accuracy | Training Time |
|-----------------|----------|---------------|
| Similar Homes   | 91.4%    | 2.3 hours     |
| Different Layouts | 84.7%  | 0.9 hours     |
| New Sensor Config | 79.3%  | 1.2 hours     |

## Installation

```
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in CSV format with timestamp, sensor ID, sensor state, and activity labels
2. Configure sensor mappings in the ArubaDatasetProcessor class
3. Run the processor to generate abstracted features
4. Train models using the provided model implementations
5. Create ensemble predictions

```python
# Example usage
data_dir = "/path/to/dataset"
raw_data_path = os.path.join(data_dir, "data.csv")

# Process data
processor = ArubaDatasetProcessor(raw_data_path)
processed_data = processor.process()

# Train models and create ensemble predictions
# (See example notebooks for complete implementation)
```

## Future Work

- Multi-resident support through personalized models
- Edge deployment for privacy and reduced latency
- Long-term drift adaptation as resident behaviors change

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code for your research, please cite:
```
Abiola, O. (2025). Human Activity Recognition in Smart Homes: A Generalizable Approach Using Functional Abstraction. Aston University, Birmingham UK.
```

## Acknowledgments

- CASAS Smart Home project for providing the datasets
- Diane Cook's work on setting-generalized activity models
