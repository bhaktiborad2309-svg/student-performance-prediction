from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import csv
import os

app = Flask(__name__)

# ✅ Load trained model + accuracy
model, accuracy = pickle.load(open('model.pkl', 'rb'))

file_name = "results.csv"

# ✅ Create CSV if not exists
if not os.path.exists(file_name):
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Hours", "Attendance", "Score", "Result"])


# ✅ Home route
@app.route('/')
def home():
    return render_template('index.html', accuracy=round(accuracy * 100, 2))


# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours = float(request.form['hours'])
        attendance = float(request.form['attendance'])
        score = float(request.form['score'])

        features = np.array([[hours, attendance, score]])

        prediction = model.predict(features)
        result = "Pass" if prediction[0] == 1 else "Fail"

        # Save to CSV
        with open(file_name, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([hours, attendance, score, result])

        return render_template(
            'index.html',
            prediction_text=f"Result: {result}",
            accuracy=round(accuracy * 100, 2)
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


# ✅ Show Data + Charts
@app.route('/data')
def show_data():
    try:
        df = pd.read_csv(file_name)

        # ---- Pass/Fail Chart ----
        result_counts = df['Result'].value_counts()
        plt.figure()
        result_counts.plot(kind='bar')
        plt.title("Pass vs Fail Distribution")
        plt.xlabel("Result")
        plt.ylabel("Number of Students")
        plt.savefig("static/chart.png")
        plt.close()

        # ---- Accuracy Chart ----
        plt.figure()
        plt.bar(['Accuracy'], [accuracy])
        plt.ylim(0, 1)
        plt.title("Model Accuracy")
        plt.savefig("static/accuracy.png")
        plt.close()

        return render_template(
            'data.html',
            table=df.to_html(classes='table table-striped', index=False),
            chart="static/chart.png",
            accuracy_chart="static/accuracy.png",
            accuracy_value=round(accuracy * 100, 2)
        )

    except Exception as e:
        return str(e)


# ✅ Download CSV
@app.route('/download')
def download_file():
    return send_file(file_name, as_attachment=True)


# ✅ Run app
if __name__ == "__main__":
    app.run()