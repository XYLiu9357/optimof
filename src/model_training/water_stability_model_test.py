import os
import joblib
from water_stability_model import WaterStabilityRF

if __name__ == "__main__":
    project_dir = "."
    rf_model_path = os.path.join(project_dir, "model", "water_rf_model.pkl")
    with open(rf_model_path, "rb") as rf_file:
        rf_model: WaterStabilityRF = joblib.load(rf_file)
    print("Random Forest Performance:")
    rf_model.run_perf_tests()

    # boost_model_path = os.path.join(project_dir, "model", "water_boost_model.pkl")
    # with open(boost_model_path, "rb") as f:
    #     boost_model: WaterStabilityRF = joblib.load(f)
    # print("Boosted Tree Performance:")
    # boost_model.run_perf_tests()
