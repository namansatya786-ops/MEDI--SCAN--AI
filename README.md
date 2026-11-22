# MediScan AI

A research-only skin lesion image classification prototype built with a small PyTorch CNN, FastAPI, and a static web interface. It displays the model's category scores; it is not a diagnostic or clinical decision tool.

## Setup

Use Python 3.10 or later. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Prepare training data

Place images in one subfolder per category. Folder names become the model's output labels:

```text
data/
  class_a/
    image-001.jpg
  class_b/
    image-002.jpg
```

Use appropriately licensed, de-identified data with reliable labels. Keep patient data private and obtain any required permissions. The example uses a deterministic random train/validation split; validation accuracy is not evidence of clinical safety or generalization.

Train and save a local model:

```powershell
python train.py --data-dir data --epochs 10
```

The default model path is `models/skin_lesion_cnn.pt`. To load a different checkpoint, set `MEDISCAN_MODEL_PATH` before starting the server.

## Run

```powershell
python -m mediscan.app
```

Open <http://127.0.0.1:8000>. Without a trained checkpoint, the server remains available but image analysis returns a clear `503` response. The application does not persist uploaded images.

## Intake form privacy and behavior

The browser form validates the requested contact and health fields locally; it does not send or save those values. Only a selected image is sent to the local `/api/predict` endpoint, and PDF reports remain in the browser and are not analyzed. The demo has no patient-record or care-provider integration. Do not enter real patient information or deploy this prototype with patient data.

## Important limitations

- Research and education only; not validated or intended for diagnosis, triage, treatment, or clinical decisions.
- Scores are model outputs, not calibrated probabilities of disease.
- Performance depends on dataset quality and may vary across populations, image devices, and acquisition conditions.
- Do not use this prototype to delay professional care. Consult a qualified healthcare professional about skin changes or concerns.
