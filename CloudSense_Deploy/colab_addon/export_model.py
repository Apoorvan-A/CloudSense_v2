"""
=======================================================
 CloudSense — Model Export Script (Run in Colab AFTER training)
 Add this cell at the END of your Run_on_Colab.ipynb
=======================================================
 This saves the trained CEEMDAN+CNN-BiLSTM model and scaler
 so you can deploy it on AWS EC2.
"""

import torch
import pickle
import json
import os
import numpy as np
from google.colab import files

# 1. Make export directory
os.makedirs("model_export", exist_ok=True)

# 2. Save the trained model weights
# 'ceemdan_model' is the variable name from main.py / train_evaluate.py
# Saves only weights (smaller file, portable)
torch.save(ceemdan_model.state_dict(), "model_export/ceemdan_bilstm.pth")
print(" Model weights saved → model_export/ceemdan_bilstm.pth")

# 3. Save model architecture config
model_config = {
    "look_back":    48,          # sequence length used during training
    "n_imfs":       6,           # number of CEEMDAN IMFs
    "hidden_size":  64,
    "conv_filters": 32,
    "num_layers":   1,
    "dropout":      0.2,
    "input_size":   1,
    "output_size":  1,
}
with open("model_export/model_config.json", "w") as f:
    json.dump(model_config, f, indent=2)
print(" Model config saved  → model_export/model_config.json")

# 4. Save the MinMaxScaler so predictions are on correct scale
with open("model_export/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(" Scaler saved        → model_export/scaler.pkl")

# 5. Save the CEEMDAN IMF decomposition of the TRAINING series
# The inference server needs the last 48 samples of each IMF to predict next step
# Save only the last 200 points (enough for a sliding-window start)
imf_tail = {str(i): imfs[i][-200:].tolist() for i in range(len(imfs))}
with open("model_export/imf_tail.json", "w") as f:
    json.dump(imf_tail, f)
print(" IMF tail saved      → model_export/imf_tail.json")

# 6. Save a quick sanity-check metric
sanity = {
    "mae":  float(results["CEEMDAN+CNN-BiLSTM (Proposed)"]["MAE"]),
    "rmse": float(results["CEEMDAN+CNN-BiLSTM (Proposed)"]["RMSE"]),
    "r2":   float(results["CEEMDAN+CNN-BiLSTM (Proposed)"]["R2"]),
}
with open("model_export/training_metrics.json", "w") as f:
    json.dump(sanity, f, indent=2)
print(" Metrics saved       → model_export/training_metrics.json")

# 7. Zip and download
import shutil
shutil.make_archive("cloudsense_model_export", "zip", "model_export")
print("\n Zipping...")
files.download("cloudsense_model_export.zip")
print("⬇  Download started — save this ZIP, you'll upload it to EC2.")
