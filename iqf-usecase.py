import os
import shutil
import piq
import torch

from glob import glob
from scipy import ndimage
from typing import Any, Dict, Optional, Union, Tuple, List

import cv2
import mlflow
import numpy as np

from iquaflow.datasets import DSWrapper
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.task_execution import PythonScriptTaskExecution
from iquaflow.metrics import SNRMetric
from iquaflow.metrics import SharpnessMetric as RERMetric

from custom_iqf import DSModifierMSRN, DSModifierFSRCNN,  DSModifierLIIF, DSModifierESRGAN
from custom_iqf import SimilarityMetrics

def rm_experiment(experiment_name = "SiSR"):
    """Remove previous mlflow records of previous executions of the same experiment"""
    try:
        mlflow.delete_experiment(ExperimentInfo(f"{experiment_name}").experiment_id)
    except:
        pass
    shutil.rmtree("mlruns/.trash/",ignore_errors=True)
    os.makedirs("mlruns/.trash/",exist_ok=True)
    shutil.rmtree(f"./Data/test-ds/.ipynb_checkpoints",ignore_errors=True)

#Define name of IQF experiment
experiment_name = "SiSR"

# Remove previous mlflow records of previous executions of the same experiment
rm_experiment(experiment_name = experiment_name)

#Define path of the original(reference) dataset
data_path = f"./Data/test-ds"

#DS wrapper is the class that encapsulate a dataset
ds_wrapper = DSWrapper(data_path=data_path)

#Define path of the training script
python_ml_script_path = 'custom_train.py'

#List of modifications that will be applied to the original dataset:

ds_modifiers_list = [
    DSModifierMSRN( params={
        'zoom':3,
        'model':"MSRN_nonoise/MSRN_1to033/model_epoch_1500.pth"
    } ),
    DSModifierLIIF( params={
        'config0':"LIIF_config.json",
        'config1':"test_liif.yaml",
        'model':"LIIF_blur/epoch-best.pth" 
    } ),
    DSModifierFSRCNN( params={
        'config':"test_scale3.json",
        'model':"FSRCNN_1to033_x3_blur/best.pth"
    } ),
    DSModifierESRGAN( params={
        'zoom':3,
        'model':"ESRGAN_1to033_x3_blur/net_g_latest.pth"
    } )
]

# Task execution executes the training loop
task = PythonScriptTaskExecution( model_script_path = python_ml_script_path )

#Experiment definition, pass as arguments all the components defined beforehand
experiment = ExperimentSetup(
    experiment_name=experiment_name,
    task_instance=task,
    ref_dsw_train=ds_wrapper,
    ds_modifiers_list=ds_modifiers_list,
    repetitions=1
)

#Execute the experiment
experiment.execute()

# ExperimentInfo is used to retrieve all the information of the whole experiment. 
# It contains built in operations but also it can be used to retrieve raw data for futher analysis

experiment_info = ExperimentInfo(experiment_name)

print('Calculating similarity metrics...')

win = 28
_ = experiment_info.apply_metric_per_run(
    SimilarityMetrics(
        experiment_info,
        n_jobs               = 15,
        ext                  = 'tif',
        n_pyramids           = 1,
        slice_size           = 7,
        n_descriptors        = win*2,
        n_repeat_projection  = win,
        proj_per_repeat      = 4,
        device               = 'cpu',
        return_by_resolution = False,
        pyramid_batchsize    = win,
        use_liif_loader      = True
    ),
    ds_wrapper.json_annotations,
)

print('Calculating RER Metric...')

_ = experiment_info.apply_metric_per_run(
    RERMetric(
        experiment_info,
        ext="tif",
        window_size=64, 
    ),
    ds_wrapper.json_annotations,
)

print('Calculating SNR Metric...')

__ = experiment_info.apply_metric_per_run(
     SNRMetric(
         experiment_info,
         ext="tif"
     ),
     ds_wrapper.json_annotations,
 )

df = experiment_info.get_df(
    ds_params=["modifier"],
    metrics=[
        'ssim','psnr','swd',"snr_median", "snr_mean", "snr_std",
        'fid','FWHM_Y', 'MTF_NYQ_X', 'MTF_halfNYQ_X', 'MTF_halfNYQ_other',
         'RER_Y', 'RER_other', 'RER_X', 'FWHM_other', 'MTF_NYQ_other'
         ],
    dropna=False
)

print(df)

df.to_csv(f'./{experiment_name}.csv')
