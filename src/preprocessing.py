"""
This is the preprocessing module.

This module contains functions that help with preprocessing of the Huggingface audio dataset.
"""

import numpy as np
from transformers import ASTFeatureExtractor
from typing import Dict, Any, List


def calculate_stats(examples: Dict[str, Dict[str, Any]], 
                    audio_field: str, 
                    array_field: str,
                    feature_extractor: ASTFeatureExtractor) -> Dict[str, List[float]]:
    """
    Calculates the mean and standard deviation of the spectrogram of the audio examples in the provided batch.

    Args:
        examples (Dict[str, Any]): A dictionary of audio examples, where each example is itself a dictionary with an audio 
            field containing an audio array, and a label field containing a label value.
        audio_field (str): The name of the field in the examples that contains the audio file information.
        array_field (str): The name of the field in the audio_fielf that contains the audio arrays.
        feature_extractor (ASTFeatureExtractor): An instance of the Hugging Face feature extractor to be used.

    Returns:
        Dict[str, List[float]]: A dictionary containing two keys: 'mean' and 'std', each with a 
        list of floats representing the corresponding statistic for each example in the dataset.
    """
    audio_arrays = [x[f"{array_field}"] - x[f"{array_field}"].mean() for x in examples[f"{audio_field}"]]
    fbanks = feature_extractor(audio_arrays, sampling_rate=feature_extractor.sampling_rate)
    
    mean = [np.mean(fbank) for fbank in fbanks['input_values']]
    std = [np.std(fbank) for fbank in fbanks['input_values']]
    
    return {'mean':mean, 'std':std}


def preprocess_audio_arrays(examples: Dict[str, Any], 
                            audio_field: str,
                            array_field: str,
                            feature_extractor: ASTFeatureExtractor) -> Dict[str, Any]:
    """
    This function takes in a dictionary of audio examples and preprocesses them using a Hugging Face feature extractor.
    
    Args:
        examples (Dict[str, Any]): A dictionary of audio examples, where each example is itself a dictionary with an audio 
            field containing an audio array, and a label field containing a label value.
        audio_field (str): The name of the audio field in the example dictionaries.
        array_field (str): The name of the audio array field within the audio field.
        feature_extractor (ASTFeatureExtractor): The Hugging Face feature extractor to use for preprocessing the audio.
    
    Returns:
        Dict[str, Any]: Dictionary containing the preprocessed audio inputs as tensors. The key of the dictionary is
            'input_values', which is a list of tensors, where each tensor is a preprocessed audio input.
    """
    
    audio_arrays = [x[f"{array_field}"] for x in examples[f"{audio_field}"]]
    outputs = feature_extractor(audio_arrays, sampling_rate=feature_extractor.sampling_rate)
    return outputs
