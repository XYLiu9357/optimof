"""app.py
Backend of Flask web app.
"""

from pathlib import Path

import numpy as np
from flask import Flask, render_template, request

from src.utils.predict import predict_from_file

project_path = Path(".")
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]

    # Check if the file ends with ".cif"
    if not file.filename.endswith(".cif"):
        return "Invalid file type. Please upload a .cif file.", 400

    # Save the file to a temporary location
    input_path = Path("app") / "temp" / file.filename
    file.save(input_path)

    # Call the prediction function and get three numpy arrays
    predictions = predict_from_file(project_path, input_path)

    # Convert the numpy arrays to lists for display purposes
    prediction_1, prediction_2, prediction_3 = [arr.tolist() for arr in predictions]

    # Render the predictions in the result template
    return render_template(
        "result.html", pred1=prediction_1, pred2=prediction_2, pred3=prediction_3
    )


def run_flask_client():
    app.run(debug=True)


if __name__ == "__main__":
    app.run(debug=True)
