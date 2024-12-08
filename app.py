from flask import Flask, request, jsonify , render_template
import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load('decision_tree_model.pkl')

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the request
        data = request.get_json()

        # Convert JSON to DataFrame
        # Ensure keys match the column names used during training
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]

        # Return the result as JSON
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability[0])
        })
    except Exception as e:
        # Handle errors and return a meaningful message
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
