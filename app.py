print("Starting app...")

from flask import Flask, render_template, request
import pickle
import traceback

print("Loading model...")
with open("spam_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)
print("Model loaded successfully!")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        message = request.form["message"]
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "🚨 This message is SPAM!"
        else:
            result = "✅ This message is NOT Spam."

    return render_template("index.html", result=result)

print("Starting Flask server...")
app.run(debug=True)