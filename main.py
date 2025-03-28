import os
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import pickle
from chatbot import CustomerRetentionChatbot

# Load environment variables and configure the page
load_dotenv()
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

def init_session_state():
    # Set up session state variables if they don't exist
    default_state = {
        'chat_history': [],
        'messages': [],
        'prediction_data': None,
        'show_prediction': False,
        'show_chatbot': False,
        'show_viz': False
    }
    
    for key, default_value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = CustomerRetentionChatbot()

def load_models():
    try:
        # Load all the necessary models and encoders
        model = load_model('model.h5')
        
        with open('onehot_encoder_geo.pkl', 'rb') as f:
            geo_encoder = pickle.load(f)
        with open('label_encoder_gender.pkl', 'rb') as f:
            gender_encoder = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        return model, geo_encoder, gender_encoder, scaler
    except Exception as e:
        st.error(f"Couldn't load models: {str(e)}")
        st.stop()

def prepare_data_for_prediction(inputs, gender_encoder):
    # Convert user inputs into model-ready format
    is_active = 1 if inputs['is_active_member'] == 'Yes' else 0
    has_card = 1 if inputs['has_cr_card'] == 'Yes' else 0
    
    # Create a dataframe with all features
    data = pd.DataFrame({
        'CreditScore': [inputs['credit_score']],
        'Gender': [gender_encoder.transform([inputs['gender']])[0]],
        'Age': [inputs['age']],
        'Tenure': [inputs['tenure']],
        'Balance': [inputs['balance']],
        'NumOfProducts': [inputs['num_of_products']],
        'HasCrCard': [has_card],
        'IsActiveMember': [is_active],
        'EstimatedSalary': [inputs['estimated_salary']],
        'Geography_France': [0],
        'Geography_Germany': [0],
        'Geography_Spain': [0]
    })
    
    # Set the geography column
    data[f'Geography_{inputs["geography"]}'] = 1
    
    return data, is_active

def predict_churn(data, model, scaler):
    try:
        # Scale the data and make prediction
        scaled_data = scaler.transform(data)
        return model.predict(scaled_data)[0][0]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

def get_sidebar_inputs(geo_encoder, gender_encoder):
    # Create a clean sidebar layout for user inputs
    st.sidebar.markdown("""
        <div style='padding: 10px; background: #f8f9fa; border-radius: 8px;'>
            <h2 style='margin: 0; font-size: 1.1rem;'>🔍 Customer Details</h2>
        </div>
    """, unsafe_allow_html=True)
    
    inputs = {}
    
    # Personal info
    st.sidebar.markdown("### 👤 Personal Information")
    inputs['geography'] = st.sidebar.selectbox('🌍 Country', geo_encoder.categories_[0])
    inputs['gender'] = st.sidebar.selectbox('👤 Gender', gender_encoder.classes_)
    inputs['age'] = st.sidebar.slider('🎂 Age', 18, 92, 18)
    
    # Financial info
    st.sidebar.markdown("### 💰 Financial Information")
    inputs['credit_score'] = st.sidebar.number_input(
        '📊 Credit Score',
        min_value=300,
        max_value=850,
        value=None,
        placeholder="Enter score (300-850)"
    )
    
    inputs['balance'] = st.sidebar.number_input(
        '💰 Account Balance',
        min_value=0,
        value=None,
        placeholder="Enter balance"
    )
    
    inputs['estimated_salary'] = st.sidebar.number_input(
        '💵 Annual Salary',
        min_value=0,
        value=None,
        placeholder="Enter annual salary"
    )
    
    # Bank relationship
    st.sidebar.markdown("### 🏦 Bank Relationship")
    inputs['tenure'] = st.sidebar.slider('📅 Years with Bank', 0, 10, 0)
    inputs['num_of_products'] = st.sidebar.slider('🛒 Bank Products', 1, 4, 1)
    inputs['has_cr_card'] = st.sidebar.selectbox('💳 Credit Card', ['No', 'Yes'])
    inputs['is_active_member'] = st.sidebar.selectbox('✅ Active Member', ['No', 'Yes'])
    
    predict_btn = st.sidebar.button("🔮 Predict Churn Risk", use_container_width=True)
    
    return inputs, predict_btn

def check_inputs(inputs):
    errors = []
    
    if not (300 <= inputs['credit_score'] <= 850):
        errors.append("Credit score should be between 300 and 850")
    if inputs['balance'] < 0:
        errors.append("Balance can't be negative")
    if inputs['estimated_salary'] < 0:
        errors.append("Salary can't be negative")
    
    return errors

def create_customer_chart(data):
    # Create a bar chart showing customer attributes
    features = ['Credit Score', 'Age', 'Balance', 'Tenure', 'Products']
    values = [
        data['credit_score'] / 1000,
        data['age'] / 100,
        data['balance'] / 100000,
        data['tenure'] / 10,
        data['num_of_products'] / 4
    ]
    
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#F44336']
    
    fig = go.Figure(data=[
        go.Bar(
            x=features,
            y=values,
            text=[f'{v:.2f}' for v in values],
            textposition='auto',
            marker_color=colors,
            hovertemplate='%{x}: %{y:.2f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Customer Profile',
        xaxis_title='Attributes',
        yaxis_title='Normalized Values',
        plot_bgcolor='white',
        showlegend=False,
        font=dict(size=14),
        xaxis=dict(tickangle=-45),
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def show_prediction():
    pred = st.session_state.prediction_data['probability']
    
    # Set up the display based on risk level
    color = "#FF5252" if pred > 0.5 else "#4CAF50"
    risk = "High Risk" if pred > 0.5 else "Low Risk"
    icon = "⚠️" if pred > 0.5 else "✅"
    
    # Show the prediction card
    st.markdown(f"""
        <div style='background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h2 style='margin-bottom: 20px; color: #333;'>Results</h2>
            <div style='background: {color}1A; padding: 20px; border-radius: 8px; border-left: 5px solid {color};'>
                <div style='display: flex; align-items: center;'>
                    <span style='font-size: 2em; margin-right: 15px;'>{icon}</span>
                    <div>
                        <h3 style='margin: 0; color: {color}; font-size: 1.5em;'>{risk}</h3>
                        <p style='margin: 5px 0 0 0; font-size: 1.2em; color: #666;'>
                            {pred:.1%} chance of churning
                        </p>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Show the chart
    if st.session_state.show_viz:
        st.markdown("""
            <div style='margin-top: 30px; background: white; padding: 30px; border-radius: 10px;'>
        """, unsafe_allow_html=True)
        fig = create_customer_chart(st.session_state.prediction_data)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

def save_prediction(pred, inputs, is_active):
    # Save everything we need for display
    st.session_state.prediction_data = {
        'probability': pred,
        'credit_score': inputs['credit_score'],
        'balance': inputs['balance'],
        'age': inputs['age'],
        'tenure': inputs['tenure'],
        'num_of_products': inputs['num_of_products'],
        'is_active_member': bool(is_active),
        'geography': inputs['geography'],
        'gender': inputs['gender']
    }
    
    st.session_state.show_prediction = True
    st.session_state.show_viz = True

def main():
    # Set up everything we need
    init_session_state()
    model, geo_encoder, gender_encoder, scaler = load_models()
    
    st.title("📊 Customer Churn Predictor")
    
    # Split the page into two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Add a professional intro
        st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3>Customer Retention Analysis</h3>
                <p>Analyze customer behavior patterns to predict potential churn risk and take proactive retention measures.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show results if we have them
        if st.session_state.show_prediction and st.session_state.prediction_data:
            show_prediction()
    
    # Get all the inputs from sidebar
    inputs, predict_clicked = get_sidebar_inputs(geo_encoder, gender_encoder)
    
    # Make prediction if requested
    if predict_clicked:
        errors = check_inputs(inputs)
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            # Prepare data and predict
            data, is_active = prepare_data_for_prediction(inputs, gender_encoder)
            pred = predict_churn(data, model, scaler)
            
            if pred is not None:
                save_prediction(pred, inputs, is_active)
                st.session_state.show_chatbot = True
                st.rerun()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Chat section with improved styling
    with col2:
        st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='margin-top: 0;'>💬 Customer Support Assistant</h3>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                    <p style='margin: 0; color: #666;'>Our AI assistant is here to help you understand the prediction results and provide retention strategies.</p>
                </div>
                <div style='background: #e9ecef; padding: 10px; border-radius: 8px; text-align: center;'>
                    <p style='margin: 0; color: #495057;'>Chat support is temporarily unavailable.</p>
                    <p style='margin: 5px 0 0 0; font-size: 0.9em; color: #6c757d;'>Please try again later.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()