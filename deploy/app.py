import os
import sys
import tempfile

from flask import Flask, flash, redirect, render_template, request, url_for

_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_DEPLOY_DIR)
sys.path.append(os.path.dirname(_DEPLOY_DIR))

import pandas as pd

from detector_runtime import Detector
from feature_contract import ContractMismatchError, window_coverage_warning
from pcap_parser import parse_pcap_to_dataframe, probe_pcap_runtime

DEFAULT_BUNDLE_DIR = "model_bundle"

# An upload the single worker can finish inside gunicorn's request timeout.
DEFAULT_MAX_UPLOAD_BYTES = 8 * 1024 * 1024
# The results page renders one row per flow, so an uncapped table turns a few
# megabytes of CSV into hundreds of megabytes of HTML.
DEFAULT_MAX_RENDERED_ROWS = 1000


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
    # A per-process random key silently invalidates every other worker's
    # session cookies, so more than one gunicorn worker loses flash state.
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24).hex()
    # Scoring and rendering are both linear in row count, and a 100 MiB CSV is
    # roughly 7.5M rows: enough to exhaust memory and kill the worker before it
    # answers. This cap is what the worker can actually finish.
    app.config['MAX_CONTENT_LENGTH'] = int(
        os.environ.get('MAX_UPLOAD_BYTES', DEFAULT_MAX_UPLOAD_BYTES)
    )
    app.config['MAX_RENDERED_ROWS'] = int(
        os.environ.get('MAX_RENDERED_ROWS', DEFAULT_MAX_RENDERED_ROWS)
    )
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
        app.config['BUNDLE_ERROR'] = None
    except ContractMismatchError as error:
        # Worth its own message: this bundle is not broken, it is out of date,
        # and the fix is to retrain rather than to check the file paths.
        app.logger.error(
            "Model bundle at %s declares an incompatible contract: %s",
            resolved_bundle_dir,
            error,
        )
        app.config['DETECTOR'] = None
        app.config['BUNDLE_ERROR'] = (
            f"The configured model bundle is not compatible with this runtime. {error}"
        )
    except Exception as error:
        # exception() keeps the traceback: without it a genuine code bug and a
        # malformed manifest produce the same one-line log.
        app.logger.exception(
            "Failed to load model bundle from %s: %s", resolved_bundle_dir, error
        )
        app.config['DETECTOR'] = None
        app.config['BUNDLE_ERROR'] = None

    @app.route('/')
    def index():
        if app.config['DETECTOR'] is None:
            flash(
                app.config.get('BUNDLE_ERROR')
                or "Detector not loaded. Check server logs for errors.",
                "danger",
            )
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
        coverage_notes: list[str] = []
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                # Bind the path before the write. Assigning after save() meant a
                # failed write left tmp_path None, so the finally block below
                # skipped cleanup and the file leaked.
                tmp_path = tmp_file.name
                file.save(tmp_path)

            if file_extension == 'csv':
                flows = pd.read_csv(tmp_path)
            else:
                flows = parse_pcap_to_dataframe(tmp_path, detector.feature_names)

            if flows.empty:
                flash("File parsed, but no valid network flows were extracted.", "warning")
                return redirect(url_for('index'))

            # Coverage is part of the answer, not a footnote. A bundle with no
            # window block cannot see rate-driven behaviour, and a capture too
            # short for the widest window produces window values that look
            # confident and mean nothing.
            if not detector.has_window_coverage:
                reason = detector.window_coverage_reason
                coverage_notes.append(
                    "This model was trained without temporal window features, so "
                    "rate-driven behaviour such as DDoS, scanning and brute force "
                    "is outside its coverage."
                    + (f" Reason recorded at training time: {reason}" if reason else "")
                )
            else:
                span_note = window_coverage_warning(flows)
                if span_note:
                    coverage_notes.append(span_note)

            detections = detector.detect(flows, explain=True)
        # UnicodeDecodeError subclasses ValueError, so it has to be caught
        # first or the generic clause below swallows it and shows a codec
        # message to someone who just picked the wrong file.
        except UnicodeDecodeError:
            flash(
                "That file is not readable as CSV text. Upload a flow CSV, or "
                "enable PCAP uploads to submit a capture.",
                "danger",
            )
            return redirect(url_for('index'))
        except ValueError as error:
            flash(f"Could not score uploaded file: {error}", "danger")
            return redirect(url_for('index'))
        except (FileNotFoundError, KeyError, RuntimeError, OSError) as error:
            flash(f"Error processing file: {error}", "danger")
            return redirect(url_for('index'))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        total_samples = int(len(detections))
        total_anomalies = int(detections['is_anomaly'].sum()) if total_samples else 0
        anomaly_frequency = (total_anomalies / total_samples * 100) if total_samples else 0.0

        # Every flow is scored and counted, but only a bounded slice is turned
        # into rows and rendered. One <tr> per flow makes the response grow
        # with the upload, so an uncapped table is how a legal upload becomes
        # hundreds of megabytes of HTML.
        limit = app.config['MAX_RENDERED_ROWS']
        shown = detections.head(limit) if total_samples > limit else detections
        results = [
            {
                'index': position + 1,
                'prediction': row.prediction,
                'confidence': (
                    'n/a' if row.prediction == 'Unknown anomaly' else f"{row.confidence:.4f}"
                ),
                'anomaly_score': f"{row.anomaly_score:.4f}",
                'is_anomaly': row.is_anomaly,
                'explanation': getattr(row, 'explanation', []) or [],
            }
            for position, row in enumerate(shown.itertuples())
        ]
        return render_template(
            'results.html',
            results=results,
            total_samples=total_samples,
            total_anomalies=total_anomalies,
            anomaly_frequency=f"{anomaly_frequency:.2f}",
            coverage_notes=coverage_notes,
            rendered_rows=len(results),
            truncated=total_samples > len(results),
        )

    return app


app = create_app(os.environ.get('MODEL_BUNDLE_DIR', DEFAULT_BUNDLE_DIR))


if __name__ == '__main__':
    app.run()
