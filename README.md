# Stock Prediction Experiments

Python experiments that train neural networks on historical stock-price features and generate forecasts and diagnostic plots. The code is intended for learning and research; its output is not validated trading performance.

## Setup

Use a Python environment compatible with the pinned NumPy, pandas, SciPy, and scikit-learn versions in `requirements.txt` (Python 3.11 is a practical starting point). TensorFlow availability depends on your platform.

```sh
git clone https://github.com/Neel-Sh/Stock-Prediction.git
cd Stock-Prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train and predict

```sh
python train_model.py
python predict_realtime.py
```

Both scripts prompt for a ticker, using `AAPL` when input is empty. Train first, then predict using the same ticker and the generated model artifacts. Training downloads market data and currently requests **100 epochs** for the ensemble; it can take significant time and resources.

## Files

| File | Purpose |
| --- | --- |
| `train_model.py` | Downloads data, computes features, prepares sequences, trains and saves models |
| `predict_realtime.py` | Loads saved artifacts and produces forecasts and charts |
| `requirements.txt` | Python dependencies |
| `model_analysis.png` | Checked-in example visualization; not a current benchmark |

Training writes ticker-prefixed `.h5` models and `.pkl` scaler, feature, and metadata files. These generated artifacts stay ignored. Only load pickle/model files you trust.

## Evaluation limits

Read preprocessing and split logic before interpreting reported accuracy: scaling is fitted during feature preparation, before the later train/test split, so holdout metrics can include information leakage. Historical fit, heuristic confidence scores, and buy/sell labels do not establish out-of-sample returns. External data availability can also change independently of this project.

See [Contributing](CONTRIBUTING.md) for development checks. No repository-wide license file is included.
