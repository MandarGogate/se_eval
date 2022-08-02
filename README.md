# se_eval - A fast wrapper around several freely available implementations of objective metrics for speech enhancement and separation


## Install with pip
```bash
pip install https://github.com/mandargogate/se_eval/archive/master.zip
```

## Objective evaluation metrics
- Perceptual Evaluation of Speech Quality (PESQ)
- Short-time objective intelligibility (STOI)
- Scale-invariant Signal-to-Distortion Ratio (SI-SDR)
- Scale-Invariant Signal-to-Noise Ratio (SI-SNR)
- Signal to Distortion Ratio (SDR)
- Signal-to-Noise Ratio (SNR)
- Composite Objective Speech Quality (composite)**
- Hearing-Aid Speech Perception Index (HASPI)***
- Hearing-Aid Speech Quality Index (HASQI)***
- Virtual Speech Quality Objective Listener (VISQOL)***


** requires pysepm ```pip3 install https://github.com/schmiph2/pysepm/archive/master.zip```

*** requires <a href="https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html" target="_blank">MATLAB Engine API for Python</a>

## Usage

See `examples/test.py`

```
Options:
  --testing_root test_root   Utterances root for noisy and enhanced
  --clean_root clean_root    Clean utterances root
  --matlab_path path         MATLAB path for HASPI, HASQI and VISQOL
  --latex False              Generate latex table for results
  --metrics stoi pesq        List of objective evaluation metrics
  --multiprocessing True     Use multiprocessing module for faster evaluation
  --fs 16000                 Audio sampling frequency
  --model_uids noisy         List of models e.g. noisy baseline dnn
```