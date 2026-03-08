# Car Insurance Premium Prediction Model

An AI-powered machine learning application that predicts car insurance premiums based on various vehicle and owner characteristics.

## Features

- **Machine Learning Models**: Trained using Gradient Boosting Regressor with 96.29% R² accuracy
- **Interactive Web App**: Built with Streamlit for easy predictions
- **Data Preprocessing**: Handles missing values and categorical encoding
- **Visualizations**: Interactive charts showing feature impacts on premiums

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jay202425/CARINSURANCE12.git
cd CARINSURANCE12
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train the Model
```bash
python train_model.py
```

### Run the Streamlit App
```bash
streamlit run app.py
```

Access the app at: `http://localhost:8502`

## Model Performance

- **Algorithm**: Gradient Boosting Regressor
- **R² Score**: 0.9629 (96.29%)
- **RMSE**: ₹3,929.20
- **MAE**: ₹3,218.95

## Input Features

1. **Car Age** (Years): 0-15
2. **Car Value** (₹): Market value of the vehicle
3. **Engine CC**: Engine displacement (800-2000cc)
4. **Fuel Type**: Petrol, Diesel, Hybrid, Electric
5. **Transmission**: Automatic or Manual
6. **Owner Age** (Years): 18-75
7. **NCB Percent** (%): No-Claim Bonus (0-50%)
8. **Accident History**: Yes/No
9. **City Tier**: Tier 1, 2, or 3

## Files

- `train_model.py`: Model training script
- `app.py`: Streamlit web application
- `best_model.pkl`: Trained model
- `scaler.pkl`: Feature scaler
- `label_encoders.pkl`: Categorical encoders
- `feature_names.pkl`: Feature names
- `requirements.txt`: Python dependencies
- `car_insurance_premium_regression_dataset.csv`: Training dataset

## Technologies Used

- Python 3.14+
- Scikit-learn (Machine Learning)
- Streamlit (Web App)
- Pandas & NumPy (Data Processing)
- Plotly (Visualizations)

## License

MIT License

## Author

Jay (jaylondonintl@gmail.com)
