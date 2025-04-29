import gradio as gr
from src.predict import predict
import logging

logging.basicConfig(level=logging.INFO)

def make_prediction(*inputs):
    try:
        # Map inputs to feature names
        feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        # Create input dictionary
        input_data = dict(zip(feature_names, inputs))
        
        # Make prediction
        result = predict(input_data)
        
        if result['error']:
            return f"Error: {result['error']}"
            
        # Format prediction result
        prediction = "Positive (Diabetic)" if result['prediction'] == 1 else "Negative (Non-diabetic)"
        probability = result['probability'] * 100
        
        return f"Prediction: {prediction}\nProbability: {probability:.2f}%"
        
    except Exception as e:
        logging.error(f"Interface error: {str(e)}")
        return f"Error: {str(e)}"

# Create interface
iface = gr.Interface(
    fn=make_prediction,
    inputs=[
        gr.Slider(0, 17, label="Pregnancies (Number of pregnancies)"),
        gr.Slider(0, 200, label="Glucose (Plasma glucose concentration mg/dl)"),
        gr.Slider(0, 122, label="BloodPressure (Diastolic blood pressure mm Hg)"),
        gr.Slider(0, 99, label="SkinThickness (Triceps skin fold thickness mm)"),
        gr.Slider(0, 846, label="Insulin (2-Hour serum insulin mu U/ml)"),
        gr.Slider(0, 67.1, label="BMI (Body mass index kg/m²)"),
        gr.Slider(0.078, 2.42, label="DiabetesPedigreeFunction (Diabetes pedigree function)"),
        gr.Slider(21, 81, label="Age (Age in years)")
    ],
    outputs="text",
    title="MedExplain: AI-Driven Medical Diagnosis Support",
    description="Transparent, interpretable medical predictions using explainable AI"
)

if __name__ == "__main__":
    iface.launch()

