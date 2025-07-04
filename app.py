from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')

# Cột đầu vào đúng chuẩn lúc train
feature_columns = [
    'Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 'Running Nose',
    'Fatigue', 'Gastrointestinal', 'Headache',
    'Asthma', 'Chronic Lung Disease', 'Heart Disease', 'Diabetes', 'Hyper Tension',
    'Abroad travel', 'Contact with COVID Patient', 'Visited Public Exposed Places',
    'Attended Large Gathering', 'Family working in Public Exposed Places',
    'Wearing Masks', 'Sanitization from Market'
]

@app.route('/')
def home():
    return render_template('index.html')  # Trang chính tổng quan

@app.route('/canhiem')
def canhiem():
    return render_template('canhiem.html')  # Trang dự đoán nguy cơ

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({'error': 'Dữ liệu gửi không hợp lệ'}), 400

        # Khởi tạo mặc định tất cả là 0
        input_data = {col: 0 for col in feature_columns}

        # Cập nhật giá trị từ client, chỉ chấp nhận 0 hoặc 1
        for key, value in data.items():
            if key in input_data:
                input_data[key] = 1 if str(value).strip() in ['1', 'True', 'true'] else 0

        # Chuyển thành DataFrame đúng thứ tự
        input_df = pd.DataFrame([input_data])[feature_columns]

        # In log để kiểm tra dữ liệu vào model
        print("\nDữ liệu truyền vào model:")
        print(input_df)

        prediction = model.predict(input_df)[0]

        result = (
            'Bạn có thể đã nhiễm COVID-19. Hãy liên hệ cơ sở y tế để xét nghiệm.'
            if prediction == 1 else
            'Nguy cơ thấp. Tuy nhiên vẫn nên theo dõi sức khỏe.'
        )

        return jsonify({'prediction': result})

    except Exception as e:
        print("Lỗi server:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
