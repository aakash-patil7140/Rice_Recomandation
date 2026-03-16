from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model_data = joblib.load('rice_recommender_model.pkl')
model = model_data['model']
le = model_data['label_encoder']
features = model_data['features']
bool_cols = model_data['bool_cols']

# Load rice nutrition data
rice_df = pd.read_csv('realistic_indian_rice_varieties_100.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET'])
def predict():
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def result():
    result = None
    nutrition = None

    try:
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        diabetes = int('diabetes' in request.form)
        anemia = int('anemia' in request.form)
        athlete = int('athlete' in request.form)
        overweight = int('overweight' in request.form)

        bmi = weight / ((height / 100) ** 2)
        bmi = round(bmi, 1)

        person_data = {
            'Age': age,
            'Weight (kg)': weight,
            'Height (cm)': height,
            'BMI': bmi,
            'Diabetes': diabetes,
            'Anemia': anemia,
            'Athlete': athlete,
            'Overweight': overweight
        }

        df = pd.DataFrame([person_data])
        for col in bool_cols:
            df[col] = df[col].astype(int)

        df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100],
                                    labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df = pd.get_dummies(df, columns=['BMI_Category'], drop_first=True)

        for feature in features:
            if feature not in df.columns:
                df[feature] = 0

        df = df[features]
        pred_label = model.predict(df)[0]
        recommended_rice = le.inverse_transform([pred_label])[0]

        nutrition_info = rice_df[rice_df['Variety Name'] == recommended_rice]
        if not nutrition_info.empty:
            nutrition = nutrition_info.iloc[0].to_dict()

        result = {
            'rice': recommended_rice,
            'bmi': bmi
        }

    except Exception as e:
        result = {'error': str(e)}

    return render_template('result.html', result=result, nutrition=nutrition)

if __name__ == '__main__':
    app.run(debug=True)



