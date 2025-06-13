from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load mô hình đã huấn luyện (gồm cả preprocessor và model bên trong pipeline)
model = joblib.load('model.pkl')

# Các cột đầu vào gốc (trùng với lúc huấn luyện)
symptom_columns = [
    'Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
    'None_Sympton', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea', 'None_Experiencing'
]
age_columns = ['Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+']
gender_columns = ['Gender_Female', 'Gender_Male', 'Gender_Transgender']
severity_columns = ['Severity_Mild', 'Severity_Moderate', 'Severity_None', 'Severity_Severe']
contact_columns = ['Contact_Dont-Know', 'Contact_No', 'Contact_Yes']

# Gộp toàn bộ cột đầu vào
all_input_columns = symptom_columns + age_columns + gender_columns + severity_columns + contact_columns + ['Country']

@app.route('/')
def home():
    return render_template('canhiem.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nếu là JSON gửi từ JS
        if request.is_json:
            data = request.get_json()
            input_df = pd.DataFrame([data])

        # Nếu là dữ liệu gửi từ form HTML
        else:
            input_data = [request.form.get(col, 0) for col in all_input_columns]
            input_data = [float(x) if x.replace('.', '', 1).isdigit() else x for x in input_data]
            input_df = pd.DataFrame([input_data], columns=all_input_columns)

        # Dự đoán trực tiếp với pipeline đã lưu (gồm preprocessor)
        prediction = model.predict(input_df)[0]
        result = 'Nguy cơ cao' if prediction == 1 else 'Nguy cơ thấp'

        return jsonify({'prediction': result})

    except Exception as e:
        print("Lỗi server:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
