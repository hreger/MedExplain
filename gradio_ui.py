import gradio as gr
from src.predict import predict_with_explanation  # Update this line
import logging

logging.basicConfig(level=logging.INFO)

def make_prediction(*inputs):
    try:
        # Map inputs to feature names
        feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        input_data = dict(zip(feature_names, inputs))
        result = predict_with_explanation(input_data)
        
        if result['error']:
            return f"Error: {result['error']}"
            
        # Format prediction result
        prediction = "Positive (Diabetic)" if result['prediction'] == 1 else "Negative (Non-diabetic)"
        probability = result['probability'] * 100
        
        # Format explanations
        explanation = f"Prediction: {prediction}\nProbability: {probability:.2f}%\n\n"
        explanation += "Top Contributing Factors:\n"
        
        # Add feature importance explanation
        for feature, importance in result['feature_importance'][:3]:
            impact = "increases" if importance > 0 else "decreases"
            explanation += f"- {feature} {impact} diabetes risk (impact: {abs(importance):.3f})\n"
        
        # Add LIME explanation
        explanation += "\nDetailed Feature Contributions (LIME):\n"
        for feat, imp in result['lime_explanation'][:3]:
            explanation += f"- {feat}: {imp:.3f}\n"
            
        return explanation
        
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

