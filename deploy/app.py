import os
import sys
import tempfile

from flask import Flask, flash, redirect, render_template, request, url_for

_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_DEPLOY_DIR)
sys.path.append(os.path.dirname(_DEPLOY_DIR))

import pandas as pd

from detector_runtime import Detector
from pcap_parser import parse_pcap_to_dataframe, probe_pcap_runtime

DEFAULT_BUNDLE_DIR = "model_bundle"


def _resolve_bundle_dir(bundle_dir: str | os.PathLike) -> str:
    bundle_path = os.fspath(bundle_dir)
    if os.path.isabs(bundle_path):
        return bundle_path
    return os.path.join(_DEPLOY_DIR, bundle_path)


def create_app(
    bundle_dir: "str | os.PathLike" = DEFAULT_BUNDLE_DIR,
    enable_pcap_uploads: bool | None = None,
) -> Flask:
    """Build the Flask app around one immutable model bundle.

    If the bundle fails to load, the app still starts (so ops can see the
    error page and logs) but `/predict` refuses to score anything instead
    of falling back to a degraded or legacy detector.
    """

    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.urandom(24).hex()
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
    resolved_bundle_dir = _resolve_bundle_dir(bundle_dir)
    app.config['MODEL_BUNDLE_DIR'] = resolved_bundle_dir
    if enable_pcap_uploads is None:
        requested = os.environ.get('ENABLE_PCAP_UPLOADS', '').strip().lower() in {'1', 'true', 'yes'}
        if requested:
            enable_pcap_uploads, reason = probe_pcap_runtime()
            if not enable_pcap_uploads:
                app.logger.warning("PCAP uploads were requested but disabled: %s", reason)
        else:
            enable_pcap_uploads = False
    app.config['PCAP_UPLOADS_ENABLED'] = enable_pcap_uploads

    try:
        app.config['DETECTOR'] = Detector(resolved_bundle_dir)
    except Exception as error:
        app.logger.error("Failed to load model bundle from %s: %s", resolved_bundle_dir, error)
        app.config['DETECTOR'] = None

    @app.route('/')
    def index():
        if app.config['DETECTOR'] is None:
            flash("Detector not loaded. Check server logs for errors.", "danger")
        return render_template(
            'index.html',
            pcap_uploads_enabled=app.config['PCAP_UPLOADS_ENABLED'],
        )

    @app.route('/predict', methods=['POST'])
    def predict():
        detector = app.config['DETECTOR']
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

        allowed_extensions = {'csv'}
        if app.config['PCAP_UPLOADS_ENABLED']:
            allowed_extensions.update({'pcap', 'pcapng'})
        file_extension = file.filename.rsplit('.', 1)[-1].lower()
        if file_extension not in allowed_extensions:
            if app.config['PCAP_UPLOADS_ENABLED']:
                flash("Unsupported file type. Please upload a CSV, PCAP, or PCAPNG file.", "warning")
            else:
                flash(
                    "Unsupported file type. This deployment accepts CSV uploads only unless PCAP uploads are explicitly enabled.",
                    "warning",
                )
            return redirect(url_for('index'))

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                file.save(tmp_file.name)
                tmp_path = tmp_file.name

            if file_extension == 'csv':
                flows = pd.read_csv(tmp_path)
            else:
                flows = parse_pcap_to_dataframe(tmp_path, detector.feature_names)

            if flows.empty:
                flash("File parsed, but no valid network flows were extracted.", "warning")
                return redirect(url_for('index'))

            detections = detector.detect(flows)
        except ValueError as error:
            flash(f"Could not score uploaded file: {error}", "danger")
            return redirect(url_for('index'))
        except (FileNotFoundError, KeyError, RuntimeError) as error:
            flash(f"Error processing file: {error}", "danger")
            return redirect(url_for('index'))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        results = [
            {
                'index': position + 1,
                'prediction': row.prediction,
                'confidence': (
                    'n/a' if row.prediction == 'Unknown anomaly' else f"{row.confidence:.4f}"
                ),
                'anomaly_score': f"{row.anomaly_score:.4f}",
                'is_anomaly': row.is_anomaly,
            }
            for position, row in enumerate(detections.itertuples())
        ]
        total_samples = len(results)
        total_anomalies = sum(1 for row in results if row['is_anomaly'])
        anomaly_frequency = (total_anomalies / total_samples * 100) if total_samples else 0.0
        flash(
            f"Processed {total_samples} samples. Found {total_anomalies} "
            f"({anomaly_frequency:.2f}%) anomalies",
            "success",
        )
        return render_template(
            'results.html',
            results=results,
            total_samples=total_samples,
            total_anomalies=total_anomalies,
            anomaly_frequency=f"{anomaly_frequency:.2f}",
        )

    return app


app = create_app(os.environ.get('MODEL_BUNDLE_DIR', DEFAULT_BUNDLE_DIR))


if __name__ == '__main__':
    app.run()
