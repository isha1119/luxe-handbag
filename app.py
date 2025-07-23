from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model and encoders
model = joblib.load("svm_brand_model.pkl")
le_sub = joblib.load("subcategory_encoder.pkl")
label_encoder = joblib.load("brand_label_encoder.pkl")

# Get list of all subcategories
subcategories = list(le_sub.classes_)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form inputs
        subcategory = request.form['subcategory']
        price = float(request.form['price'])
        rating = float(request.form['rating'])

        # Encode the subcategory
        sub_encoded = le_sub.transform([subcategory])[0]

        # Prepare data for prediction
        user_input = np.array([[sub_encoded, price, rating]])

        # Predict the brand
        prediction = model.predict(user_input)[0]
        predicted_brand = label_encoder.inverse_transform([prediction])[0]

        # Get prediction probabilities for both brands
        probs = model.predict_proba(user_input)[0]
        brand_probs = {
            label_encoder.inverse_transform([i])[0]: round(p * 100, 2)
            for i, p in enumerate(probs)
        }

        # Show result page with prediction and probability
        return render_template("thankyou.html", prediction=predicted_brand, probabilities=brand_probs)

    return render_template("index.html", subcategories=subcategories)

@app.route('/thankyou')
def thankyou():
    return render_template("thankyou.html")

if __name__ == '__main__':
    app.run(debug=True)
