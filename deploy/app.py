from flask import Flask, render_template, request, redirect, url_for, flash
import os
import sys
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from anomaly_detector_service import ConvAutoencoder, EXPECTED_FEATURES_CICFLOWMETER
from pcap_parser import parse_pcap_to_dataframe

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

detector = None

def load_detector():
    global detector
    if detector is None:
        try:
            detector = ConvAutoencoder(feature_columns=EXPECTED_FEATURES_CICFLOWMETER)
            print("AnomalyDetector initialized successfully for Flask app.")
        except Exception as e:
            print(f"Error initializing AnomalyDetector: {e}. Please ensure 'working/' directory and models exist.")
            detector = None # Ensure detector is None if init fails

load_detector()

@app.route('/')
def index():
    if detector is None:
        flash("Detector not loaded. Check server logs for errors.", "danger")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if detector is None:
        flash("Detector not available. Cannot perform prediction.", "danger")
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file part', 'warning')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'warning')
        return redirect(url_for('index'))

    file_extension = file.filename.split('.')[-1].lower()

    if file_extension not in ['csv', 'pcap', 'pcapng']:
        flash("Unsupported file type. Please upload a CSV, PCAP, or PCAPNG file.", "warning")
        return redirect(url_for('index'))

    tmp_pcap_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            file.save(tmp_file.name)
            tmp_pcap_path = tmp_file.name

        pcap_output_dir = os.path.join(app.root_path, 'temp_pcap_output')
        os.makedirs(pcap_output_dir, exist_ok=True)
        df_to_predict = parse_pcap_to_dataframe(tmp_pcap_path, pcap_output_dir, EXPECTED_FEATURES_CICFLOWMETER)

        if df_to_predict.empty:
            flash("PCAP file parsed, but no valid network flows were extracted.", "warning")
            return redirect(url_for('index'))

        # Ensure all expected feature columns are present, fill missing ones with 0 if they were not in the parsed data
        # This is important if the PCAP parser doesn't output all 78 features for some reason.
        for col in EXPECTED_FEATURES_CICFLOWMETER:
            if col not in df_to_predict.columns:
                df_to_predict[col] = 0.0 # Fill missing columns with 0

        # Call the predict method from your AnomalyDetector
        predictions, anomaly_scores = detector.predict(df_to_predict)

        results = []
        total_anomalies = 0
        total_samples = len(predictions)
        for i in range(total_samples):
            if predictions[i] == 1: # 1 for anomaly
                total_anomalies += 1
            results.append({
                'index': i + 1,
                'prediction': 'Anomaly' if predictions[i] == 1 else 'Benign',
                'score': f"{anomaly_scores[i]:.4f}"
            })
        anomaly_frequency = (total_anomalies / total_samples * 100) if total_samples > 0 else 0.0
        flash(f"Processed {total_samples} samples. Found {total_anomalies} ({anomaly_frequency:.2f}%) anomalies", "success")
        return render_template('results.html',
                               results=results,
                               total_samples=total_samples,
                               total_anomalies=total_anomalies ,
                               anomaly_frequency=f"{anomaly_frequency:.2f}")
    except Exception as e:
        flash(f"Error processing file: {e}", "danger")
        return redirect(url_for('index'))
    finally:
        if tmp_pcap_path and os.path.exists(tmp_pcap_path):
            os.remove(tmp_pcap_path)

    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists('working'):
        os.makedirs('working')
        print("Created 'working/' directory. Please ensure models and scaler are in it.")

    app.run()