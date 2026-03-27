import scripting
from core.pipelines import run_anomaly_detection_pipeline

if __name__ == "__main__":
    scripting.logged_main(
        "Detect anomalies in radargrams",
        run_anomaly_detection_pipeline,
        setup_run=False,
        run_topic="anomaly_detection"
    )
