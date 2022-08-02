from collections import OrderedDict
from functools import partial
from multiprocessing import Pool

import librosa
import numpy as np
import torch
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
from torchmetrics.functional.audio import signal_distortion_ratio
from torchmetrics.functional.audio import signal_noise_ratio
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from tqdm import tqdm


def multicore_processing(func, parameters: list, processes=None):
    pool = Pool(processes=processes)
    result = pool.map(func, parameters)
    pool.close()
    pool.join()
    return result


def get_metrics(combined_utterances, metric_func, multiprocessing=False, processes=None):
    if multiprocessing:
        scores = multicore_processing(metric_func, combined_utterances, processes=processes)
    else:
        scores = []
        for utterance in tqdm(combined_utterances):
            scores += [metric_func(utterance)]
    return scores


def torch_eval(combined_utterance, fs, metric_func):
    degraded, reference = combined_utterance
    target = torch.from_numpy(librosa.load(reference, sr=fs)[0])
    preds = torch.from_numpy(librosa.load(degraded, sr=fs)[0])
    assert len(target) == len(preds), "{} should have same length".format(combined_utterance)
    return metric_func(preds, target).numpy()


def composite_eval(combined_utterance, fs, metric_func):
    degraded, reference = combined_utterance
    target = librosa.load(reference, sr=fs)[0]
    preds = librosa.load(degraded, sr=fs)[0]
    assert len(target) == len(preds), "{} should have same length".format(combined_utterance)
    return metric_func(target, preds)


def get_metric_func(metric, fs=16000, pesq_mode="wb", extended_stoi=False):
    if metric == "pesq":
        return partial(torch_eval, fs=fs, metric_func=partial(perceptual_evaluation_speech_quality, fs=fs, mode=pesq_mode))
    elif metric == "stoi":
        return partial(torch_eval, fs=fs, metric_func=partial(short_time_objective_intelligibility, fs=fs, extended=extended_stoi))
    elif metric == "sisdr":
        return partial(torch_eval, fs=fs, metric_func=scale_invariant_signal_distortion_ratio)
    elif metric == "sisnr":
        return partial(torch_eval, fs=fs, metric_func=scale_invariant_signal_noise_ratio)
    elif metric == "sdr":
        return partial(torch_eval, fs=fs, metric_func=signal_distortion_ratio)
    elif metric == "snr":
        return partial(torch_eval, fs=fs, metric_func=signal_noise_ratio)
    elif metric == "composite":
        import pysepm as pm
        return partial(composite_eval, fs=fs, metric_func=partial(pm.composite, fs=fs))
    else:
        raise RuntimeError("{} not implemented".format(metric))


def get_matlab_engine(script_path):
    import matlab.engine
    matlab_engine = matlab.engine.start_matlab()
    matlab_engine.addpath(script_path, nargout=0)
    return matlab_engine


def get_matlab_metric(degraded, reference, metric, matlab_engine=None):
    if matlab_engine is None:
        close_engine = True
        matlab_engine = get_matlab_engine()
    else:
        close_engine = False

    if metric.lower() == "visqol":
        score_val = float(matlab_engine.visqol(reference, degraded, 'NB', 0, 0))
    elif metric.lower() == "hasqi":
        score_val = float(matlab_engine.HASQI_v2(reference, degraded))
    elif metric.lower() == "haspi":
        score_val = float(matlab_engine.HASPI_v2(reference, degraded))
    else:
        raise Exception("{} not implemented".format(metric))
    if close_engine:
        matlab_engine.close()
    return score_val


def calculate_matlab_metric(combined_utterances, metric, script_path):
    engine = get_matlab_engine(script_path)
    scores = []
    for combined_utterance in combined_utterances:
        degraded, reference = combined_utterance
        scores.append(get_matlab_metric(degraded, reference, metric, engine))
    engine.close()
    return np.mean(scores)


def get_se_metric(metrics, utterance_pairs, fs, multiprocessing, cores, matlab_path):
    mean_score = OrderedDict()
    for metric in metrics:
        if metric.lower() in ["pesq", "stoi", "sisdr", "sisnr", "sdr", "snr"]:
            mean_score[metric.lower()] = np.mean(get_metrics(utterance_pairs, get_metric_func(metric.lower(), fs=fs), multiprocessing=multiprocessing, processes=cores))
        elif metric.lower() == "composite":
            composite_metrics = get_metrics(utterance_pairs, get_metric_func(metric.lower(), fs=fs), multiprocessing=multiprocessing, processes=cores)
            csig, cbak, covl = np.mean(composite_metrics, axis=0)
            mean_score["csig"] = csig
            mean_score["cbak"] = cbak
            mean_score["covl"] = covl
        elif metric.lower() in ["haspi", "hasqi", "visqol"]:
            mean_score[metric.lower()] = calculate_matlab_metric(utterance_pairs, metric, matlab_path)
        else:
            raise RuntimeError("{} not implemented".format(metric))
    return mean_score
