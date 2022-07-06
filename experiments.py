import os
import shutil
import piq
import torch
import mlflow
import tempfile

from glob import glob
from scipy import ndimage
from typing import Any, Dict, Optional, Union, Tuple, List

import cv2
import numpy as np

from iquaflow.datasets import DSWrapper, DSModifier
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.task_execution import PythonScriptTaskExecution
from iquaflow.metrics import RERMetric, SNRMetric

from custom_iqf import SimilarityMetrics

def download_and_prepare_gt(resstr="033",sufix=''):

    bucket_name = "image-quality-framework"
    url = f"https://{bucket_name}.s3-eu-west-1.amazonaws.com/iq-sisr-use-case/datasets/GT_{resstr}{sufix}.zip"

    gtdir = f"./Data_{resstr}/GT_{resstr}{sufix}"

    os.system( f"wget {url} -O filename.zip" )
    os.system( f"chmod 775 filename.zip" )

    os.makedirs( gtdir, exist_ok=True )

    os.system( f"unzip -o -q filename.zip -d {gtdir}")
    os.system( f"rm filename.zip" )

    src_img_dir = os.path.join( gtdir , os.listdir(gtdir)[0] )
    dst_img_dir = os.path.join( gtdir , "images" )

    shutil.rmtree(f"{dst_img_dir}", ignore_errors=True)
    os.system( f"mv {src_img_dir} {dst_img_dir}" )

    annots_fn = os.path.join( gtdir , 'annotations.json' )
    os.system( f"touch {annots_fn}" )

    print(gtdir, os.listdir(gtdir), 'Num of GT images:', len(os.listdir(os.path.join(gtdir,'images'))))

class DSModifierFake(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder.
    This modifier copies images from a folder already preexecuted (premodified).

    Args:
        ds_modifer: DSModifier. Composed modifier child

    Attributes:
        name: str. Name of the modifier
        images_dir: str. Directory of images to copy from.
        src_ext : str = 'tif'. Extension of reference GT images
        dst_ext : str = 'tif'. Extension of images to copy from.
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
        
    """
    def __init__(
        self,
        name: str,
        zip_bucket_filename: str,
        src_ext : str = 'tif',
        dst_ext : str = 'tif',
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "zoom": 3
        }
    ):
        self.bucket_name            = "image-quality-framework"
        self.src_ext                = src_ext
        self.dst_ext                = dst_ext
        self.zip_bucket_filename    = zip_bucket_filename
        self.name                   = name
        self.params: Dict[str, Any] = params
        self.ds_modifier            = ds_modifier
        self.params.update({"modifier": "{}".format(self.name)})
        
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        
        ### Download src files from bucket

        with tempfile.TemporaryDirectory() as tmpdirname:

            local_fn  = os.path.join(tmpdirname,'file.zip')

            url = f"https://{self.bucket_name}.s3-eu-west-1.amazonaws.com/iq-sisr-use-case/datasets/{self.zip_bucket_filename}"

            os.system( f"wget {url} -O {local_fn}" )
            os.system( f"unzip -q {local_fn} -d {tmpdirname}")
            
            self.images_dir = os.path.join(
                tmpdirname,
                [el for el in os.listdir(tmpdirname) if el!='file.zip'][0]
                )

            ###

            input_name = os.path.basename(data_input)
            dst = os.path.join(mod_path, input_name)
            
            os.makedirs(dst, exist_ok=True)
            
            print(f'For each image file in <{data_input}>...')
            
            for image_file in glob( os.path.join(data_input,'*.'+self.src_ext) ):
                
                imgp = self._mod_img( image_file )
                cv2.imwrite( os.path.join(dst, os.path.basename(image_file)), imgp )
            
            print('Done.')
            
            return input_name

    def _mod_img(self, image_file: str) -> np.array:
        
        fn = [
            fn for fn in glob(os.path.join(self.images_dir,'*.'+self.dst_ext))
            if os.path.basename(image_file).split('.')[0]==os.path.basename(fn).split('.')[0]
        ][0]
        
        rec_img = cv2.imread(fn)
        
        return rec_img

def rm_experiment(experiment_name):
    """Remove previous mlflow records of previous executions of the same experiment"""
    try:
        mlflow.delete_experiment(ExperimentInfo(experiment_name).experiment_id)
    except:
        pass
    shutil.rmtree("mlruns/.trash/",ignore_errors=True)
    os.makedirs("mlruns/.trash/",exist_ok=True)

def execute_experiment( zip_bucket_filename_lst, resstr="033", sufix='' ):
    """
    """
    # Define name of IQF experiment
    experiment_name = f"exp -> {resstr}{sufix}"

    # Remove previous mlflow records of previous executions of the same experiment
    rm_experiment( experiment_name )

    # Define path of the original(reference) dataset
    data_path = f"./Data_{resstr}/GT_{resstr}{sufix}"
    
    #DS wrapper is the class that encapsulate a dataset
    ds_wrapper = DSWrapper(data_path=data_path)

    #Define path of the training script
    python_ml_script_path = 'custom_train.py'

    #List of modifications that will be applied to the original dataset:

    ds_modifiers_list = [
        DSModifierFake(
            zip_bucket_filename.split('.')[0],
            zip_bucket_filename,
            src_ext = 'tif',
            dst_ext = '*',
            ds_modifier= None,
            params = {
                "zoom": 3
                })
        for zip_bucket_filename in zip_bucket_filename_lst
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

    win = 128
    _ = experiment_info.apply_metric_per_run(
        SimilarityMetrics(
            experiment_info,
            n_jobs               = 20,
            img_dir_gt           = 'images',
            ext                  = 'tif',
            n_pyramids           = 2,
            slice_size           = 7,
            n_descriptors        = win*2,
            n_repeat_projection  = win,
            proj_per_repeat      = 4,
            device               = 'cpu',
            return_by_resolution = False,
            pyramid_batchsize    = win,
            use_liif_loader      = False
        ),
        ds_wrapper.json_annotations,
    )

    print('Calculating RER Metric...')

    _ = experiment_info.apply_metric_per_run(
        RERMetric(
            experiment_info,
            win=16,
            stride=16,
            ext="tif",
            n_jobs=20
        ),
        ds_wrapper.json_annotations,
    )

    print('Calculating SNR Metric...')

    __ = experiment_info.apply_metric_per_run(
        SNRMetric(
            experiment_info,
            n_jobs=20,
            ext="tif",
            patch_sizes=[30],
            confidence_limit=50.0
        ),
        ds_wrapper.json_annotations,
    )

    df = experiment_info.get_df(
        ds_params=["modifier"],
        metrics=['ssim','psnr','swd','snr','fid','rer_0','rer_1','rer_2'],
        dropna=False
    )

    print("\n\n************************************\n\n")
    print(df)
    print("\n\n************************************\n\n")

    df.to_csv(f'./exp{resstr}{sufix}.csv')

for resstr in [
    "03",
    "033",
    "05",
    "07"
]:

    for enu,zip_bucket_filename_lst in enumerate([
        [
            f'LIIF_1to{resstr}.zip'
        ],
        [
            f'ESRGAN_1to{resstr}.zip',
            f'FSRCNN_1to{resstr}.zip',
            f'MSRN_1to{resstr}.zip'
        ]
    ]):

        download_and_prepare_gt(resstr=resstr,sufix=('_LIIF' if enu==0 else ''))#_LIIF

        print('\n\n=============================================\n')
        print(f"EXECUTING EXPERIMENT WITH RES {resstr}...")
        print('\n=============================================\n')

        execute_experiment(
            zip_bucket_filename_lst,
            resstr=resstr,
            sufix=('_LIIF' if enu==0 else '')
        )#_LIIF