import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# -------------------------
# 0. Custom CSS for Styling
# -------------------------
def load_css():
    st.markdown("""
        <style>
            /* Main Header Styling */
            .css-10trblm {
                color: #B30000;
                text-align: center;
                font-size: 2.5em;
                font-weight: bold;
            }
            /* Subheader for Sections */
            h3 {
                color: #0077B6;
                border-bottom: 2px solid #eee;
                padding-bottom: 5px;
            }
            /* Prediction Result Box (Expander) */
            .stExpander {
                border: 2px solid #ddd;
                border-radius: 10px;
                padding: 10px;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            }
            /* Metric labels */
            [data-testid="stMetricLabel"] {
                font-size: 1.1em;
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)

# -------------------------
# 1. Load Model Pipeline
# -------------------------
@st.cache_resource
def load_model(path):
    """Loads the entire pipeline (preprocessor + model)"""
    if not os.path.exists(path):
        st.error(f"Error: Model file not found at '{path}'. Please ensure 'telco_churn_voting_model.pkl' is in the same directory.")
        st.stop()
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data.get('model_pipeline') 
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load Model
model_pipeline = load_model("telco_churn_voting_model.pkl")

# -------------------------
# 2. Feature Definitions and Descriptions
# -------------------------

# Total 20 features (Must match the training order)
feature_names = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
    'MonthlyCharges', 'TotalCharges'
]

# English labels for the UI
feature_details = {
    "gender": "Customer Gender",
    "SeniorCitizen": "Is Senior Citizen? (0=No, 1=Yes)",
    "Partner": "Has a Partner?",
    "Dependents": "Has Dependents?",
    "tenure": "Tenure (Months with Company)",
    "PhoneService": "Has Phone Service?",
    "MultipleLines": "Has Multiple Lines?",
    "InternetService": "Internet Service Type",
    "OnlineSecurity": "Has Online Security?",
    "OnlineBackup": "Has Online Backup?",
    "DeviceProtection": "Has Device Protection?",
    "TechSupport": "Has Tech Support?",
    "StreamingTV": "Has Streaming TV?",
    "StreamingMovies": "Has Streaming Movies?",
    "Contract": "Contract Type",
    "PaperlessBilling": "Paperless Billing?",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges ($)",
    "TotalCharges": "Total Charges ($)",
}

# Dropdown options 
dropdown_options = {
    'gender': ('Male', 'Female'),
    'SeniorCitizen': (0, 1),
    'Partner': ('Yes', 'No'),
    'Dependents': ('Yes', 'No'),
    'PhoneService': ('Yes', 'No'),
    'MultipleLines': ('No phone service', 'No', 'Yes'),
    'InternetService': ('DSL', 'Fiber optic', 'No'),
    'OnlineSecurity': ('No internet service', 'No', 'Yes'),
    'OnlineBackup': ('No internet service', 'No', 'Yes'),
    'DeviceProtection': ('No internet service', 'No', 'Yes'),
    'TechSupport': ('No internet service', 'No', 'Yes'),
    'StreamingTV': ('No internet service', 'No', 'Yes'),
    'StreamingMovies': ('No internet service', 'No', 'Yes'),
    'Contract': ('Month-to-month', 'One year', 'Two year'),
    'PaperlessBilling': ('Yes', 'No'),
    'PaymentMethod': ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')
}

# --- Main App Function ---
def main():
    load_css()
    st.title("📞 Telco Customer Churn Risk Assessment")
    st.markdown("---")
    st.write("Enter customer details below to assess the probability of them leaving the company. **(Powered by Voting Ensemble Classifier)**")

    input_dict = {}

    # --- Section 1: General Profile ---
    st.markdown("### 1. General Profile & Contract")
    
    col1, col2 = st.columns(2)

    with col1:
        # Profile Details
        input_dict['gender'] = st.selectbox(feature_details['gender'], dropdown_options['gender'])
        input_dict['SeniorCitizen'] = st.selectbox(feature_details['SeniorCitizen'], dropdown_options['SeniorCitizen'])
        input_dict['Partner'] = st.selectbox(feature_details['Partner'], dropdown_options['Partner'])
        input_dict['Dependents'] = st.selectbox(feature_details['Dependents'], dropdown_options['Dependents'])
        input_dict['PaperlessBilling'] = st.selectbox(feature_details['PaperlessBilling'], dropdown_options['PaperlessBilling'])
    
    with col2:
        # Financial & Contract Details
        input_dict['tenure'] = st.slider(feature_details['tenure'], 0, 72, 24)
        input_dict['Contract'] = st.selectbox(feature_details['Contract'], dropdown_options['Contract'])
        input_dict['PaymentMethod'] = st.selectbox(feature_details['PaymentMethod'], dropdown_options['PaymentMethod'])
        input_dict['MonthlyCharges'] = st.number_input(feature_details['MonthlyCharges'], 18.0, 150.0, 70.0, step=0.5)
        input_dict['TotalCharges'] = st.number_input(feature_details['TotalCharges'], 0.0, 10000.0, 1500.0, step=10.0)

    st.markdown("---")
    
    # --- Section 2: Services ---
    st.markdown("### 2. Service Subscriptions")
    
    service_cols = st.columns(3)
    service_features = [
        'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    for i, feature in enumerate(service_features):
        with service_cols[i % 3]:
            input_dict[feature] = st.selectbox(feature_details[feature], dropdown_options[feature])

    # -------------------------
    # Predict Button
    # -------------------------
    st.markdown("---")
    center_col = st.columns(3)[1]

    with center_col:
        predict_btn = st.button("🔍 Predict Churn Risk", use_container_width=True)

    if predict_btn:
        # Convert input dict to DataFrame
        input_df = pd.DataFrame([input_dict], columns=feature_names)
        
        # Prediction
        pred = model_pipeline.predict(input_df)[0]
        prob_churn = model_pipeline.predict_proba(input_df)[0][1] # Probability of Churn (1)
        
        st.markdown("---")
        st.subheader("🔎 Prediction Results")

        # --- Risk Tier & Deadline Logic ---
        if prob_churn >= 0.65:
            risk_tier = "CRITICAL HIGH RISK 🔴"
            retention_deadline = "Within 48 Hours"
            action_plan = "Immediate intervention required. Assign Senior Manager and offer personalized retention plan."
        elif prob_churn >= 0.35:
            risk_tier = "MODERATE RISK 🟠"
            retention_deadline = "Next 7 Days"
            action_plan = "Send proactive offers (discounts/upgrades) to test loyalty and satisfaction."
        else:
            risk_tier = "LOW RISK 🟢"
            retention_deadline = "No Urgent Deadline"
            action_plan = "Continue standard monitoring. Review again in 3 months."

        # --- Display Output ---
        
        if pred == 1 or prob_churn >= 0.35:
            # We treat moderate risk (0.35+) as a warning state
            st.error(f"🚨 {risk_tier}: Customer is predicted to **CHURN**.")
            
            with st.expander("⏳ **RETENTION ACTION TIMELINE**", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(label="Churn Probability", value=f"{prob_churn*100:.2f}%")
                with c2:
                    st.metric(label="Action Deadline", value=retention_deadline)
                
                st.markdown(f"**Recommended Action:** {action_plan}")
                st.markdown("---")
                st.markdown("### 👉 Verdict: Should this customer switch?")
                st.markdown(f"**Analysis:** The model indicates a **{risk_tier}**. If the company fails to improve services or offer incentives by the deadline, the customer is **likely to switch** to a better provider.")
            
        else:
            st.success(f"💚 {risk_tier}: Customer is **LOYAL**.")
            
            with st.expander("✅ **MONITORING TIMELINE**", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(label="Churn Probability", value=f"{prob_churn*100:.2f}%")
                with c2:
                    st.metric(label="Review Schedule", value="Standard Cycle")

                st.markdown(f"**Recommended Action:** {action_plan}")
                st.markdown("---")
                st.markdown("### 👉 Verdict: Should this customer switch?")
                st.markdown("**Analysis:** The customer appears satisfied. There is no immediate reason for them to switch providers at this time.")

if __name__ == '__main__':
    main()