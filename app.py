import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Car Insurance Premium Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.3em;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 30px;
    }
    .input-section {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .result-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load saved models and encoders
@st.cache_resource
def load_models():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, encoders, features

model, scaler, encoders, feature_names = load_models()

# Header
st.markdown('<p class="main-header">🚗 Car Insurance Premium Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Insurance Premium Estimation</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Info", "📈 Analysis"])

with tab1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("Enter Car Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        car_age = st.slider("Car Age (Years)", 0, 15, 5)
        car_value = st.number_input("Car Value (₹)", min_value=100000, value=1500000, step=100000)
        engine_cc = st.selectbox("Engine CC", [800, 1000, 1200, 1500, 2000])
        fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel", "hybrid", "electric"])
        
    with col2:
        transmission = st.selectbox("Transmission", ["automatic", "manual"])
        owner_age = st.slider("Owner Age (Years)", 18, 75, 40)
        ncb_percent = st.slider("NCB Percent (%)", 0, 50, 20)
        accident_history = st.selectbox("Accident History", ["no", "yes"])
        city_tier = st.selectbox("City Tier", ["tier1", "tier2", "tier3"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prepare input data
    if st.button("🔮 Predict Premium", use_container_width=True, type="primary"):
        # Encode categorical variables
        try:
            fuel_encoded = encoders['fuel_type'].transform([fuel_type])[0]
            trans_encoded = encoders['transmission'].transform([transmission])[0]
            acc_encoded = encoders['accident_history'].transform([accident_history])[0]
            city_encoded = encoders['city_tier'].transform([city_tier])[0]
            
            # Create feature vector
            input_data = np.array([[
                car_age,
                car_value,
                engine_cc,
                fuel_encoded,
                trans_encoded,
                owner_age,
                ncb_percent,
                acc_encoded,
                city_encoded
            ]])
            
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            # Display result
            st.markdown(f"""
                <div class="result-box">
                    <h2 style="color: #28a745; margin: 0;">Estimated Annual Insurance Premium</h2>
                    <h1 style="color: #FF6B6B; text-align: center; margin: 20px 0;">₹ {prediction:,.2f}</h1>
                    <p style="text-align: center; color: #666;">Based on your vehicle and personal information</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Breakdown
            st.subheader("📋 Summary of Input")
            summary_data = {
                "Feature": [
                    "Car Age", "Car Value", "Engine CC", "Fuel Type",
                    "Transmission", "Owner Age", "NCB %", "Accident History", "City Tier"
                ],
                "Value": [
                    f"{car_age} years", f"₹{car_value:,}", f"{engine_cc}cc", fuel_type,
                    transmission, f"{owner_age} years", f"{ncb_percent}%", accident_history, city_tier
                ]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

with tab2:
    st.subheader("📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Model Type:** Ensemble Learning (Gradient Boosting/Random Forest)
        
        **Input Features:** 9
        - Car Age
        - Car Value
        - Engine CC
        - Fuel Type
        - Transmission Type
        - Owner Age
        - NCB Percentage
        - Accident History
        - City Tier
        """)
    
    with col2:
        st.success("""
        **Performance Metrics:**
        - Trained on comprehensive insurance dataset
        - Handles missing values with median/mode imputation
        - Features scaled for optimal performance
        - Categorical variables encoded for ML compatibility
        """)
    
    st.subheader("Feature Descriptions")
    features_info = {
        "Car Age": "Age of the vehicle in years (0-15)",
        "Car Value": "Market value of the car in Rupees",
        "Engine CC": "Engine displacement in cubic centimeters",
        "Fuel Type": "Type of fuel (Petrol, Diesel, Hybrid, Electric)",
        "Transmission": "Type of transmission (Automatic, Manual)",
        "Owner Age": "Age of the car owner in years",
        "NCB Percent": "No-Claim Bonus percentage (0-50%)",
        "Accident History": "Previous accident history (Yes/No)",
        "City Tier": "Tier of the city (Tier 1, 2, or 3)"
    }
    
    for feature, description in features_info.items():
        st.markdown(f"**{feature}:** {description}")

with tab3:
    st.subheader("📈 Model Analytics")
    
    # Create sample predictions for visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Impact of Car Age on Premium**")
        ages = np.arange(0, 16)
        base_sample = np.array([[5, 1500000, 1500, 1, 0, 40, 20, 0, 1]])
        age_predictions = []
        
        for age in ages:
            sample = base_sample.copy()
            sample[0, 0] = age
            sample_scaled = scaler.transform(sample)
            pred = model.predict(sample_scaled)[0]
            age_predictions.append(pred)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=ages, y=age_predictions, mode='lines+markers', 
                                   name='Premium', line=dict(color='#FF6B6B', width=3)))
        fig1.update_layout(xaxis_title="Car Age (Years)", yaxis_title="Premium (₹)",
                          hovermode='x unified', height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("**Impact of Owner Age on Premium**")
        owner_ages = np.arange(20, 76, 5)
        owner_predictions = []
        
        for owner_age in owner_ages:
            sample = base_sample.copy()
            sample[0, 5] = owner_age
            sample_scaled = scaler.transform(sample)
            pred = model.predict(sample_scaled)[0]
            owner_predictions.append(pred)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=owner_ages, y=owner_predictions, mode='lines+markers',
                                   name='Premium', line=dict(color='#4ECDC4', width=3)))
        fig2.update_layout(xaxis_title="Owner Age (Years)", yaxis_title="Premium (₹)",
                          hovermode='x unified', height=400)
        st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🚗 Car Insurance Premium Predictor | Powered by Machine Learning</p>
        <p style="font-size: 0.9em;">This is an AI-powered tool for estimation purposes only.</p>
    </div>
""", unsafe_allow_html=True)
