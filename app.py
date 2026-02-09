# ==============================
# User-Friendly Diabetes Health Check
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Diabetes Health Check",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

df = load_data()

# -------------------------------
# Train Models
# -------------------------------
@st.cache_resource
def train_models(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Models
    logreg = LogisticRegression(random_state=42, max_iter=1000).fit(X_scaled, y)
    dtree = DecisionTreeClassifier(random_state=42).fit(X_scaled, y)
    rforest = RandomForestClassifier(random_state=42, n_estimators=100).fit(X_scaled, y)

    models = {
        "Smart Analysis (AI)": rforest,  # Put best model first
        "Pattern Recognition": logreg,
        "Decision Tree": dtree
    }

    return models, scaler, X, y

models, scaler, X, y = train_models(df)

# -------------------------------
# Helper Functions
# -------------------------------
def get_health_tip(feature, value):
    """Provide health tips based on values"""
    tips = {
        'Glucose': {
            'high': "💡 **Tip:** High glucose levels may indicate diabetes risk. Consider consulting a healthcare provider.",
            'normal': "✅ **Great!** Your glucose level is in a healthy range.",
            'low': "⚠️ **Note:** Low glucose can cause dizziness. Maintain regular meals."
        },
        'BMI': {
            'high': "💡 **Tip:** A BMI over 30 is considered obese. Regular exercise and balanced diet can help.",
            'normal': "✅ **Excellent!** Your BMI is in a healthy range.",
            'low': "⚠️ **Note:** Low BMI might indicate underweight. Consider nutritious meals."
        },
        'BloodPressure': {
            'high': "💡 **Tip:** High blood pressure increases health risks. Regular monitoring is important.",
            'normal': "✅ **Perfect!** Your blood pressure is healthy.",
            'low': "⚠️ **Note:** Very low blood pressure can cause fatigue."
        }
    }
    
    if feature == 'Glucose':
        if value > 140:
            return tips['Glucose']['high']
        elif value < 70:
            return tips['Glucose']['low']
        else:
            return tips['Glucose']['normal']
    elif feature == 'BMI':
        if value > 30:
            return tips['BMI']['high']
        elif value < 18.5:
            return tips['BMI']['low']
        else:
            return tips['BMI']['normal']
    elif feature == 'BloodPressure':
        if value > 90:
            return tips['BloodPressure']['high']
        elif value < 60:
            return tips['BloodPressure']['low']
        else:
            return tips['BloodPressure']['normal']
    return ""

# -------------------------------
# Header
# -------------------------------
st.markdown('<h1 class="main-header">🏥 Diabetes Health Check</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get an instant health assessment based on your personal information</p>', unsafe_allow_html=True)

# Important Disclaimer
st.markdown("""
<div class="warning-box">
    <h3>⚠️ Important Medical Disclaimer</h3>
    <p>This tool provides <strong>educational estimates only</strong> and is NOT a substitute for professional medical advice. 
    Always consult with a qualified healthcare provider for proper diagnosis and treatment.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar for User Input
# -------------------------------
st.sidebar.markdown("## 📋 Enter Your Health Information")
st.sidebar.markdown("*Adjust the sliders below with your health data*")

def user_input():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Personal Information")
    
    age = st.sidebar.slider(
        "**Age** (years)", 
        min_value=1, 
        max_value=120, 
        value=30,
        help="Your current age"
    )
    
    pregnancies = st.sidebar.slider(
        "**Number of Pregnancies**", 
        min_value=0, 
        max_value=20, 
        value=0,
        help="How many times you've been pregnant (for women only)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🩺 Health Measurements")
    
    glucose = st.sidebar.slider(
        "**Blood Sugar Level** (mg/dL)", 
        min_value=0, 
        max_value=200, 
        value=120,
        help="Normal fasting glucose: 70-100 mg/dL"
    )
    
    blood_pressure = st.sidebar.slider(
        "**Blood Pressure** (mm Hg)", 
        min_value=0, 
        max_value=150, 
        value=70,
        help="Normal diastolic pressure: 60-80 mm Hg"
    )
    
    bmi = st.sidebar.slider(
        "**Body Mass Index (BMI)**", 
        min_value=10.0, 
        max_value=70.0, 
        value=25.0,
        help="Healthy BMI range: 18.5-24.9"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Additional Medical Data")
    st.sidebar.caption("*Optional but helpful for accuracy*")
    
    skin_thickness = st.sidebar.slider(
        "**Skin Thickness** (mm)", 
        min_value=0, 
        max_value=100, 
        value=20,
        help="Triceps skin fold thickness"
    )
    
    insulin = st.sidebar.slider(
        "**Insulin Level** (mu U/ml)", 
        min_value=0, 
        max_value=900, 
        value=79,
        help="2-Hour serum insulin level"
    )
    
    diabetes_pedigree = st.sidebar.slider(
        "**Family History Score**", 
        min_value=0.0, 
        max_value=2.5, 
        value=0.5,
        help="Higher values indicate stronger family history of diabetes"
    )
    
    return pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    }), {
        'age': age,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'bmi': bmi
    }

input_df, raw_values = user_input()

# -------------------------------
# Display Input Summary
# -------------------------------
st.markdown("---")
st.subheader("📊 Your Health Profile Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Age", value=f"{raw_values['age']} years")
with col2:
    glucose_val = raw_values['glucose']
    glucose_delta = "Normal" if 70 <= glucose_val <= 100 else "Check"
    st.metric(label="Blood Sugar", value=f"{glucose_val} mg/dL", delta=glucose_delta)
with col3:
    bmi_val = raw_values['bmi']
    bmi_delta = "Healthy" if 18.5 <= bmi_val <= 24.9 else "Review"
    st.metric(label="BMI", value=f"{bmi_val:.1f}", delta=bmi_delta)
with col4:
    bp_val = raw_values['blood_pressure']
    bp_delta = "Good" if 60 <= bp_val <= 80 else "Monitor"
    st.metric(label="Blood Pressure", value=f"{bp_val} mm Hg", delta=bp_delta)

# Health Tips
st.markdown("### 💡 Personalized Health Insights")
col_tip1, col_tip2 = st.columns(2)
with col_tip1:
    st.info(get_health_tip('Glucose', raw_values['glucose']))
with col_tip2:
    st.info(get_health_tip('BMI', raw_values['bmi']))

# -------------------------------
# Make Predictions
# -------------------------------
st.markdown("---")
st.markdown("## 🎯 Your Diabetes Risk Assessment")

# Get predictions from all models
predictions_list = []
for name, model in models.items():
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]
    predictions_list.append({
        'model': name,
        'prediction': pred,
        'confidence': max(prob) * 100,
        'prob_diabetic': prob[1] * 100,
        'prob_healthy': prob[0] * 100
    })

# Show consensus result prominently
consensus = sum([p['prediction'] for p in predictions_list]) >= 2
consensus_confidence = np.mean([p['confidence'] for p in predictions_list])

if consensus == 1:
    st.markdown(f"""
    <div class="danger-box">
        <h2 style="color: #dc3545;">⚠️ Higher Risk Detected</h2>
        <p style="font-size: 1.2rem;">Based on the information provided, our analysis suggests you may be at <strong>higher risk for diabetes</strong>.</p>
        <p style="font-size: 1.1rem;">Average Confidence: <strong>{consensus_confidence:.1f}%</strong></p>
        <p><strong>⚕️ Next Steps:</strong></p>
        <ul>
            <li>Schedule an appointment with your doctor for proper testing</li>
            <li>Consider a fasting blood glucose test</li>
            <li>Discuss lifestyle modifications with a healthcare provider</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="success-box">
        <h2 style="color: #28a745;">✅ Lower Risk Indicated</h2>
        <p style="font-size: 1.2rem;">Based on the information provided, our analysis suggests you are at <strong>lower risk for diabetes</strong>.</p>
        <p style="font-size: 1.1rem;">Average Confidence: <strong>{consensus_confidence:.1f}%</strong></p>
        <p><strong>💚 Keep It Up:</strong></p>
        <ul>
            <li>Maintain a healthy diet and regular exercise</li>
            <li>Monitor your blood sugar levels periodically</li>
            <li>Stay within a healthy weight range</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Detailed breakdown
with st.expander("📈 See Detailed Analysis from Each Model"):
    for pred_data in predictions_list:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        
        with col_a:
            st.markdown(f"**{pred_data['model']}**")
        
        with col_b:
            if pred_data['prediction'] == 1:
                st.markdown(f"🔴 **Higher Risk**")
            else:
                st.markdown(f"🟢 **Lower Risk**")
        
        with col_c:
            st.markdown(f"Confidence: **{pred_data['confidence']:.1f}%**")
        
        # Progress bars for probabilities
        st.progress(pred_data['prob_healthy'] / 100)
        st.caption(f"Healthy: {pred_data['prob_healthy']:.1f}% | At Risk: {pred_data['prob_diabetic']:.1f}%")
        st.markdown("---")

# -------------------------------
# What Factors Matter Most?
# -------------------------------
st.markdown("---")
st.markdown("## 🔍 What Influences Your Risk?")

st.markdown("""
Understanding which health factors contribute most to diabetes risk can help you focus on what matters. 
Here's what our analysis found:
""")

# Use Random Forest for feature importance (most reliable)
rforest_model = models["Smart Analysis (AI)"]
feature_importance = pd.DataFrame({
    'Factor': X.columns,
    'Importance': rforest_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Rename features to be more user-friendly
feature_names = {
    'Glucose': '🍬 Blood Sugar Level',
    'BMI': '⚖️ Body Mass Index',
    'Age': '📅 Age',
    'DiabetesPedigreeFunction': '👨‍👩‍👧 Family History',
    'Pregnancies': '🤰 Number of Pregnancies',
    'Insulin': '💉 Insulin Level',
    'BloodPressure': '❤️ Blood Pressure',
    'SkinThickness': '📏 Skin Thickness'
}

feature_importance['Factor'] = feature_importance['Factor'].map(feature_names)

# Show top 5 factors
st.markdown("### Top 5 Most Important Factors:")
for idx, row in feature_importance.head(5).iterrows():
    percentage = row['Importance'] * 100
    st.markdown(f"**{row['Factor']}** – {percentage:.1f}% importance")
    st.progress(row['Importance'])

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn_r(feature_importance['Importance'] / feature_importance['Importance'].max())
feature_importance.set_index('Factor')['Importance'].plot(kind='barh', color=colors, ax=ax)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Health Factors Ranked by Importance', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

st.info("💡 **What this means:** Focus on the top factors to reduce your diabetes risk most effectively!")

# -------------------------------
# How Accurate Is This?
# -------------------------------
with st.expander("🎓 How Accurate Are These Predictions?"):
    st.markdown("""
    ### Understanding Model Performance
    
    Our prediction system uses three different AI models to analyze your data. Here's how well they perform:
    """)
    
    selected_model_name = st.selectbox(
        "Choose a model to see its accuracy:", 
        list(models.keys()),
        index=0
    )
    selected_model = models[selected_model_name]
    
    # Calculate metrics
    y_pred = selected_model.predict(scaler.transform(X))
    cm = confusion_matrix(y, y_pred)
    
    # Calculate accuracy metrics in simple terms
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum() * 100
    
    st.markdown(f"### {selected_model_name}")
    st.markdown(f"**Overall Accuracy:** {accuracy:.1f}% of predictions were correct")
    
    col_metric1, col_metric2 = st.columns(2)
    
    with col_metric1:
        st.metric(
            label="Healthy People Identified Correctly",
            value=f"{cm[0,0]} out of {cm[0,0] + cm[0,1]}"
        )
    
    with col_metric2:
        st.metric(
            label="At-Risk People Identified Correctly",
            value=f"{cm[1,1]} out of {cm[1,0] + cm[1,1]}"
        )
    
    # Confusion matrix visualization
    st.markdown("#### Performance Breakdown:")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False,
                xticklabels=['Predicted Healthy', 'Predicted At-Risk'],
                yticklabels=['Actually Healthy', 'Actually At-Risk'])
    ax2.set_title('Confusion Matrix: Actual vs Predicted', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    
    st.caption("""
    **How to read this:** 
    - Top-left (blue): People who are healthy and correctly identified
    - Bottom-right (darker blue): People at risk and correctly identified
    - Other boxes: Incorrect predictions (we want these to be small!)
    """)
    
    # ROC-AUC explanation
    if hasattr(selected_model, "predict_proba"):
        roc_auc = roc_auc_score(y, selected_model.predict_proba(scaler.transform(X))[:,1])
        st.markdown(f"""
        **Model Reliability Score:** {roc_auc*100:.1f}%
        
        This score tells us how good the model is at separating healthy people from those at risk. 
        A score of 100% would be perfect, and anything above 70% is considered good.
        """)

# -------------------------------
# Download Report
# -------------------------------
st.markdown("---")
st.markdown("## 📥 Download Your Health Report")

# Prepare comprehensive report
report_df = input_df.copy()

# Add user-friendly column names
report_df.columns = [
    'Number of Pregnancies',
    'Blood Sugar (mg/dL)',
    'Blood Pressure (mm Hg)',
    'Skin Thickness (mm)',
    'Insulin (mu U/ml)',
    'Body Mass Index',
    'Family History Score',
    'Age (years)'
]

# Add predictions
for pred_data in predictions_list:
    model_name = pred_data['model']
    result = "Higher Risk" if pred_data['prediction'] == 1 else "Lower Risk"
    report_df[f"{model_name} - Assessment"] = result
    report_df[f"{model_name} - Confidence"] = f"{pred_data['confidence']:.1f}%"

# Add timestamp
from datetime import datetime
report_df['Report Generated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Convert to CSV
csv = report_df.to_csv(index=False).encode('utf-8')

col_down1, col_down2 = st.columns([1, 3])

with col_down1:
    st.download_button(
        label="📄 Download Full Report",
        data=csv,
        file_name=f'diabetes_health_report_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
        help="Download a CSV file with all your data and predictions"
    )

with col_down2:
    st.info("💾 Save this report to track your health over time or share with your doctor")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h3>🏥 About This Tool</h3>
    <p>This diabetes risk assessment tool uses machine learning algorithms trained on health data to provide estimates. 
    It analyzes patterns in your health information to suggest whether you might be at higher or lower risk for diabetes.</p>
    
    <p><strong>Remember:</strong> This is an educational tool only. For accurate diagnosis and medical advice, 
    please consult with a qualified healthcare professional.</p>
    
    <p style="margin-top: 1rem; font-size: 0.9rem; color: #666;">
    📧 Questions or feedback? This tool is for informational purposes only.
    </p>
</div>
""", unsafe_allow_html=True)