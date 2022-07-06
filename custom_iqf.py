import os
import tempfile
import sys
import json
import cv2
import piq
import torch
import yaml
import signal
import time
import math

import numpy as np
import PIL.Image as pil_image

from glob import glob
from torch.utils.data import DataLoader
from typing import Any, Dict, Optional, List,Union,Tuple
from iquaflow.datasets import DSModifier
from iquaflow.metrics import Metric
from iquaflow.experiments import ExperimentInfo

from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context

import torch.backends.cudnn as cudnn

from lowresgen import LRSimulator
from torchvision import transforms
import kornia

# MSRN
from msrn.msrn import load_msrn_model, process_file_msrn

# FSRCNN
from utils.utils_fsrcnn import convert_ycbcr_to_rgb, preprocess
from models.model_fsrcnn import FSRCNN

# LIIF
from datasets.liif import datasets as datasets_liif
from utils import utils_liif
from models.liif import models as models_liif

# ESRGAN
import esrgan
from models.esrgan import RRDBNet_arch as arch

# Metrics
from swd import SlicedWassersteinDistance

class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    print("ALARM signal received")
    raise TimeOutException()

class Args():
    def __init__(self):
        pass
    
class ModelConfS3Loader():
    
    def __init__(
        self,
        model_fn      = "single_frame_sr/SISR_MSRN_X2_BICUBIC.pth",
        config_fn_lst = [],
        bucket_name   = "image-quality-framework",
        algo          = "FSRCNN",
        zoom          = 3
    ):
        
        self.fn_dict = {
            f"conf{enu}":fn
            for enu,fn in enumerate(config_fn_lst)
        }
        
        self.fn_dict["model"]   =  model_fn
        self.model_fn           =  model_fn
        self.config_fn_lst      =  config_fn_lst
        self.bucket_name        =  bucket_name
        self.algo               =  algo
        self.zoom               =  3
        
    def load_ai_model_and_stuff(self) -> List[Any]:
        
        # Rename to full bucket subdirs...
        # First file is the model
        
        print( self.fn_dict["model"] )
        
        fn_lst = [
            os.path.join("iq-sisr-use-case/models/weights",self.fn_dict["model"])
        ] + [
            os.path.join("iq-sisr-use-case/models/config",fn)
            for fn in self.config_fn_lst
        ]
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            
            print( self.fn_dict )
            
            for k in self.fn_dict:

                bucket_fn = self.fn_dict[k]

                local_fn = os.path.join(
                    tmpdirname , k
                )
                
                kind = ('weights' if k=='model' else 'config')
                
                url = f"https://{self.bucket_name}.s3-eu-west-1.amazonaws.com/iq-sisr-use-case/models/{kind}/{bucket_fn}"

                print( url , ' ' , local_fn )

                os.system( f"wget {url} -O {local_fn}" )

                if not os.path.exists( local_fn ):
                    print( 'AWS model file not found' )
                    raise

            if self.algo=='FSRCNN':

                args = self._load_args(os.path.join(
                    tmpdirname ,"conf0"
                ))
                model = self._load_model_fsrcnn(os.path.join(
                    tmpdirname ,"model"
                ), args )

            elif self.algo=='LIIF':
                
                args = self._load_args(os.path.join(
                    tmpdirname ,"conf0"
                ))
                model = self._load_model_liif(os.path.join(
                    tmpdirname ,"model"
                ), args, os.path.join(
                    tmpdirname ,"conf1"
                ) )
                
            elif self.algo=='MSRN':
                
                args = None
                model = self._load_model_msrn(
                    os.path.join(tmpdirname ,"model"),
                    n_scale=self.zoom
                )

            elif self.algo=='ESRGAN':
                
                args = esrgan.get_args( self.zoom )
                model = self._load_model_esrgan(os.path.join(
                    tmpdirname ,"model"
                ) )

            else:
                print(f"Error: unknown algo: {self.algo}")
                raise

            return model , args
    
    def _load_args( self, config_fn: str ) -> Any:
        """Load Args"""

        args = Args()
        
        with open(config_fn) as jsonfile:
            args_dict = json.load(jsonfile)

        for k in args_dict:
            setattr( args, k, args_dict[k] )

        return args
    
    def _load_model_fsrcnn( self, model_fn: str,args: Any ) -> Any:
        """Load Model"""

        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = FSRCNN(scale_factor=args.scale).to(device)

        state_dict = model.state_dict()
        for n, p in torch.load(model_fn, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

        model.eval()

        return model
    
    def _load_model_liif(self,model_fn: str,args: Any,yml_fn: str) -> Any:
        """Load Model"""
        
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        
        with open(yml_fn, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        spec = config['test_dataset']
        # Save config and spec in self
        self.config = config
        self.spec = spec
        
        model_spec = torch.load(model_fn)['model']
        model = models_liif.make(model_spec, load_sd=True).cuda()
        
        model.eval()
        
        return model
    
    def _load_model_msrn(self,model_fn: str,n_scale:int = 3) -> Any:
        """Load MSRN Model"""
        return load_msrn_model(model_fn,n_scale=n_scale)

    def _load_model_esrgan(self,model_fn: str) -> Any:
        """Load ESRGAN Model"""
        device = torch.device('cuda')
        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        #model.load_state_dict(torch.load(args.model_path), strict=True)
        weights = torch.load(model_fn)
        model.load_state_dict(weights['params'])
        model.eval()
        model = model.to(device)
        return model

#########################
# Custom IQF
#########################

class DSModifierLIIF(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder.

    Args:
        ds_modifer: DSModifier. Composed modifier child

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo":"LIIF",
            "config0": "LIIF_config.json",
            "config1": "test_liif.yaml",
            "model": "liif_UCMerced/epoch-best.pth"
        },
    ):
        params['algo'] = 'LIIF'
        algo           = params['algo']
        
        subname = algo + '_' + \
                    os.path.splitext(params['config0'])[0]+ \
                    '_' + os.path.splitext(params['config1'])[0]+ \
                    '_'+os.path.splitext(params['model'])[0].replace('/','-')
        
        self.name = f"sisr+{subname}"
        
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})
        
        model_conf = ModelConfS3Loader(
                model_fn      = params['model'],
                config_fn_lst = [params['config0'],params['config1']],
                bucket_name   = "image-quality-framework",
                algo          = "LIIF"
        )
        
        model,args = model_conf.load_ai_model_and_stuff()
        
        self.spec   =  model_conf.spec
        self.args   =  args
        self.model  =  model
        
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        """Modify images
        Iterates the data_input path loading images, processing with _mod_img(), and saving to mod_path
        Args
            data_input: str. Path of the original folder containing images
            mod_path: str. Path to the new dataset
        Returns:
            Name of the new folder containign the images
        """
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        
        print(f'For each image file in <{data_input}>...')
        
        data_norm, eval_type, eval_bsize = None,None, None
        
        spec = self.spec
        
        spec['batch_size'] = 1
        
        spec['dataset'] = {
            'name': 'image-folder',
            'args': {
                'root_path': data_input
            }
        }
        
        dataset = datasets_liif.make(spec['dataset'])
        dataset = datasets_liif.make(spec['wrapper'], args={'dataset': dataset})
        loader = DataLoader(dataset, batch_size=spec['batch_size'],shuffle=False,
                           num_workers=1, pin_memory=False ) #pin_memory=True
        
        if data_norm is None:
        
            data_norm = {
                'inp': {'sub': [0], 'div': [1]},
                'gt': {'sub': [0], 'div': [1]}
            }
        
        inp_sub = torch.FloatTensor(data_norm['inp']['sub']).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(data_norm['inp']['div']).view(1, -1, 1, 1).cuda()
        gt_sub  = torch.FloatTensor(data_norm['gt']['sub']).view(1, 1, -1).cuda()
        gt_div  = torch.FloatTensor(data_norm['gt']['div']).view(1, 1, -1).cuda()

        if eval_type is None:
            metric_fn = utils_liif.calc_psnr
        elif eval_type.startswith('div2k'):
            scale = int(eval_type.split('-')[1])
            metric_fn = partial(utils_liif.calc_psnr, dataset='div2k', scale=scale)
        elif eval_type.startswith('benchmark'):
            scale = int(eval_type.split('-')[1])
            metric_fn = partial(utils_liif.calc_psnr, dataset='benchmark', scale=scale)
        else:
            raise NotImplementedError

        val_res  = utils_liif.Averager()
        val_ssim = utils_liif.Averager() #test

        #pbar = tqdm(loader, leave=False, desc='val')
        image_file_lst = sorted( glob( os.path.join(data_input,'*.tif') ) )
        count = 0
        for enu,batch in enumerate(loader):
            
            image_file = image_file_lst[enu]
            
            try:
                imgp = self._mod_img( batch, inp_sub, inp_div, eval_bsize,gt_div , gt_sub )
                print(imgp.shape)
                output = pil_image.fromarray((imgp*255).astype(np.uint8))
                output.save( os.path.join(dst, os.path.basename(image_file)) )
                #cv2.imwrite( os.path.join(dst, os.path.basename(image_file)), imgp )
            except Exception as e:
                print(e)
            
        return input_name

    def _mod_img(self, batch: Any, inp_sub: Any, inp_div: Any, eval_bsize: Any, gt_div: Any, gt_sub: Any) -> None:

        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        
        if eval_bsize is None:
            with torch.no_grad():
                pred = self.model(inp, batch['coord'], batch['cell'])
        else:
            pred = batched_predict(self.model, inp,
                                   batch['coord'], batch['cell'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)
        ih, iw = batch['inp'].shape[-2:]
        s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
        shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
        pred = pred.view(*shape) \
            .permute(0, 1, 2, 3).contiguous()
        return pred.detach().cpu().numpy().squeeze()

class DSModifierFSRCNN(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder.

    Args:
        ds_modifer: DSModifier. Composed modifier child

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo":"FSRCNN",
            "config": "test.json",
            "model": "FSRCNN_1to033_x3_noblur/best.pth"
        },
    ):
        
        params['algo'] = 'FSRCNN'
        algo           = params['algo']
        
        subname = algo + '_' + os.path.splitext(params['config'])[0]+'_'+os.path.splitext(params['model'])[0].replace('/','-')
        self.name = f"sisr+{subname}"
        
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})

        model_conf = ModelConfS3Loader(
                model_fn      = params['model'],
                config_fn_lst = [params['config']],
                bucket_name   = "image-quality-framework",
                algo          = "FSRCNN"
        )
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model, self.args = model_conf.load_ai_model_and_stuff()
        
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        """
        Modify images
        Iterates the data_input path loading images, processing with _mod_img(), and saving to mod_path

        Args
            data_input: str. Path of the original folder containing images
            mod_path: str. Path to the new dataset
        Returns:
            Name of the new folder containign the images
        """
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        
        print(f'For each image file in <{data_input}>...')
        
        for image_file in glob( os.path.join(data_input,'*.tif') ):

            try:
                imgp = self._mod_img( image_file )
                print( imgp.shape )
                output = pil_image.fromarray(imgp)
                output.save(os.path.join(dst, os.path.basename(image_file)))
                #cv2.imwrite( os.path.join(dst, os.path.basename(image_file)), imgp )
            except Exception as e:
                print(e)
        
        print('Done.')
        
        return input_name

    def _mod_img(self, image_file: str) -> np.array:
        
        args = self.args
        model = self.model

        image = pil_image.open(image_file).convert('RGB')

        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale

        # AFEGIT PER FER EL BLUR
        img_tensor = transforms.ToTensor()(image).unsqueeze_(0)
        sigma = 0.5 * args.scale
        kernel_size = math.ceil(sigma * 3 + 4)
        kernel_tensor = kornia.filters.get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
        image_blur = kornia.filter2d(img_tensor, kernel_tensor[None])
        image = transforms.ToPILImage()(image_blur.squeeze_(0))
        ########

        hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        
        bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)

        lr, _ = preprocess(lr, self.device)
        _, ycbcr = preprocess(bicubic, self.device)

        with torch.no_grad():
            preds = model(lr).clamp(0.0, 1.0)

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)

        return output

class DSModifierMSRN(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder.

    Args:
        ds_modifer: DSModifier. Composed modifier child

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo":"MSRN",
            "zoom": 3,
            "model": "MSRN/SISR_MSRN_X2_BICUBIC.pth"
        },
    ):
        
        params['algo'] = 'MSRN'
        algo           = params['algo']
        subname = algo + '_'+os.path.splitext(params['model'])[0].replace('/','-')
        self.name = f"sisr+{subname}"
        
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})
        
        model_conf = ModelConfS3Loader(
                model_fn      = params['model'],
                config_fn_lst = [],
                bucket_name   = "image-quality-framework",
                algo          = "MSRN",
                zoom          = self.params['zoom']
        )
        
        self.model,_ = model_conf.load_ai_model_and_stuff()
        
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        """Modify images
        Iterates the data_input path loading images, processing with _mod_img(), and saving to mod_path

        Args
            data_input: str. Path of the original folder containing images
            mod_path: str. Path to the new dataset
        Returns:
            Name of the new folder containign the images
        """
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        
        print(f'For each image file in <{data_input}>...')
        
        for image_file in glob( os.path.join(data_input,'*.tif') ):
            
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(5)
            
            try:
                imgp = self._mod_img( image_file )
                print( imgp.shape )
                cv2.imwrite( os.path.join(dst, os.path.basename(image_file)), imgp )
            except TimeOutException as ex:
                print(ex)
            except Exception as e:
                print(e)
            
            signal.alarm(0)
        
        print('Done.')
        
        return input_name

    def _mod_img(self, image_file: str) -> np.array:
        
        zoom       = self.params["zoom"]
        loaded     = cv2.imread(image_file, -1)
        wind_size  = loaded.shape[1]
        gpu_device = "0"
        res_output = 1/zoom # inria resolution

        rec_img = process_file_msrn(
            loaded,
            self.model,
            compress=True,
            out_win = loaded.shape[-2],
            wind_size=wind_size+10, stride=wind_size+10,
            scale=3,
            batch_size=1,
            padding=5
        )
        
        return rec_img


class DSModifierESRGAN(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder.

    Args:
        ds_modifer: DSModifier. Composed modifier child

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo":"ESRGAN",
            "zoom": 3,
            "model": "./ESRGAN_1to033_x3_blur/net_g_latest.pth"
        },
    ):
        
        params['algo']              = 'ESRGAN'
        algo                        = params['algo']
        subname                     = algo + '_'+os.path.splitext(params['model'])[0].replace('/','-')
        self.name                   = f"sisr+{subname}"
        self.params: Dict[str, Any] = params
        self.ds_modifier            = ds_modifier
        self.device                 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.params.update({"modifier": "{}".format(self._get_name())})

        model_conf = ModelConfS3Loader(
            model_fn      = params['model'],
            config_fn_lst = [],
            bucket_name   = "image-quality-framework",
            algo          = "ESRGAN"
        )
        
        self.model , self.args = model_conf.load_ai_model_and_stuff()
        
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        """Modify images
        Iterates the data_input path loading images, processing with _mod_img(), and saving to mod_path

        Args
            data_input: str. Path of the original folder containing images
            mod_path: str. Path to the new dataset
        Returns:
            Name of the new folder containign the images
        """
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        
        print(f'For each image file in <{data_input}>...')
        
        for image_file in glob( os.path.join(data_input,'*.tif') ):
            
            imgp = self._mod_img( image_file )
            print( imgp.shape )
            cv2.imwrite( os.path.join(dst, os.path.basename(image_file)), imgp )
        
        print('Done.')
        
        return input_name

    def _mod_img(self, image_file: str) -> np.array:

        img = esrgan.generate_lowres( image_file , scale=self.args.zoom )

        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)
        
        with torch.no_grad():

            output = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        
        rec_img = (output*255).astype(np.uint8)

        return rec_img

#########################
# Fake Modifier
#########################

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
        images_dir: str,
        src_ext : str = 'tif',
        dst_ext : str = 'tif',
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "zoom": 3
        }
    ):
        self.src_ext                = src_ext
        self.dst_ext                = dst_ext
        self.images_dir             = images_dir
        self.name                   = name
        self.params: Dict[str, Any] = params
        self.ds_modifier            = ds_modifier
        self.params.update({"modifier": "{}".format(self.name)})
        
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        
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

#########################
# Similarity Metrics
#########################
    
class SimilarityMetrics( Metric ):
    
    def __init__(
        self,
        experiment_info: ExperimentInfo,
        img_dir_gt: str = "test",
        n_jobs: int = 20,
        ext: str = 'tif',
        n_pyramids:Union[int, None]=None,
        slice_size:int=7,
        n_descriptors:int=128,
        n_repeat_projection:int=128,
        proj_per_repeat:int=4,
        device:str='cpu',
        return_by_resolution:bool=False,
        pyramid_batchsize:int=128,
        use_liif_loader : bool = True
    ) -> None:
        
        self.img_dir_gt = img_dir_gt
        self.n_jobs     = n_jobs
        self.ext        = ext
        
        self.metric_names = [
            'ssim',
            'psnr',
            #'gmsd',
            #'mdsi',
            #'haarpsi',
            'swd',
            'fid'
        ]
        self.experiment_info = experiment_info
        
        self.n_pyramids           = n_pyramids
        self.slice_size           = slice_size
        self.n_descriptors        = n_descriptors
        self.n_repeat_projection  = n_repeat_projection
        self.proj_per_repeat      = proj_per_repeat
        self.device               = device
        self.return_by_resolution = return_by_resolution
        self.pyramid_batchsize    = pyramid_batchsize

        self.use_liif_loader      = use_liif_loader
    
    def _liff_loader_first_time(self,data_input:str) -> None:
    
        dsm_liif = DSModifierLIIF( params={
                'config0':"LIIF_config.json",
                'config1':"test_liif.yaml",
                'model':"LIIF_blur/epoch-best.pth" 
            } )
                
        spec = dsm_liif.spec

        spec['batch_size'] = 1

        spec['dataset'] = {
            'name': 'image-folder',
            'args': {
                'root_path': data_input
            }
        }

        self.spec = spec
        
        dataset = datasets_liif.make(spec['dataset'])
        dataset = datasets_liif.make(spec['wrapper'], args={'dataset': dataset})

        self.dataset = dataset

    def _gen_liif_loader(self):

        loader = DataLoader(self.dataset, batch_size=self.spec['batch_size'],shuffle=False,
                num_workers=1, pin_memory=True, multiprocessing_context=get_context('loky'))
        
        return loader

    def _parallel(self, fid:object,swdobj:object, pred_fn:str) -> List[Dict[str,Any]]:

        # pred_fn be like: xview_id1529imgset0012+hrn.png
        img_name = os.path.basename(pred_fn)
        gt_fn = os.path.join(self.data_path,self.img_dir_gt,f"{img_name}")

        #import pdb; pdb.set_trace()

        if self.use_liif_loader and 'LIIF' in pred_fn:
            
            # pred

            pred = cv2.imread( pred_fn )/255

            pred = pred[...,::-1].copy()

            pred = torch.from_numpy( pred )
            pred = pred.view(1,-1,pred.shape[-2],pred.shape[-1])
            pred = torch.transpose( pred, 3, 1 )

            # gt
            
            loader = self._gen_liif_loader()

            enu_file = [
                enu for enu,fn in enumerate(
                    sorted(glob(
                        os.path.join(self.data_path,self.img_dir_gt,'*')
                        ))
                    )
                if os.path.basename(fn)==img_name
                ][0]

            batch = [batch for enu,batch in enumerate(loader) if enu==enu_file][0]

            gt = torch.clamp( batch['gt'] , min=0.0, max=1.0 )

            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [round(ih * s), round(iw * s), 3]
            gt = torch.clamp( batch['gt'].view(1, *shape).transpose(3,1), min=0.0, max=1.0 )

        elif not self.use_liif_loader and 'LIIF' in pred_fn:
            
            pred = cv2.imread( pred_fn )/255
            gt = cv2.imread( gt_fn )/255

            #     pred = pred[...,::-1].copy()
            
            pred = torch.from_numpy( pred )
            gt = torch.from_numpy( gt )

            pred = pred.view(1,-1,pred.shape[-2],pred.shape[-1])
            gt = gt.view(1,-1,gt.shape[-2],gt.shape[-1])
            
            pred = torch.transpose( pred, 3, 1 )
            gt   = torch.transpose( gt  , 3, 1 )

        else:

            pred = transforms.ToTensor()(pil_image.open(pred_fn).convert('RGB')).unsqueeze_(0)
            image = pil_image.open(gt_fn).convert('RGB')
            img_tensor = transforms.ToTensor()(image).unsqueeze_(0)
            scale = 3
            sigma = 0.5*scale
            kernel_size = 9
            kernel_tensor = kornia.filters.get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
            gt = torch.clamp( kornia.filter2d(img_tensor, kernel_tensor[None]), min=0.0, max=1.0 )
        
        results_dict = {
            "ssim":None,
            "psnr":None,
            "swd":None,
            "fid":None
        }

        if pred.size()!=gt.size():
            
            #print('different size found', pred.size(), gt.size())
            
            pred_for_metrics = torch.clamp( LRSimulator(None,3)._resize(
                    pred,
                    gt.shape[-2],
                    interpolation = 'bicubic',
                    align_corners = True,
                    side = "short",
                    antialias = False
                ), min=0.0, max=1.0 )
            
            if pred_for_metrics.size()!=gt.size():
                return results_dict

        else:

            pred_for_metrics = pred
        
        results_dict = {
            "ssim":piq.ssim(pred_for_metrics,gt).item(),
            "psnr":piq.psnr(pred_for_metrics,gt).item(),
            "fid":np.sum( [
                fid( torch.squeeze(pred_for_metrics)[i,...], torch.squeeze(gt)[i,...] ).item()
                for i in range( pred_for_metrics.shape[1] )
                ] ) / pred_for_metrics.shape[1]
        }
        
        # Make gt even
        if gt.shape[-2]%2!=0 : # gt size is odd

            gt_for_swd = torch.clamp( LRSimulator(None,3)._resize(
                    gt,
                    gt.shape[-2]+1,
                    interpolation = 'bicubic',
                    align_corners = True,
                    side = "short",
                    antialias = False
                ), min=0.0, max=1.0 )

        else:

            gt_for_swd = gt
            
        # Pred should be same as GT
        if pred.size()!=gt_for_swd.size():
            
            pred_for_swd = torch.clamp( LRSimulator(None,3)._resize(
                    pred,
                    gt_for_swd.shape[-2],
                    interpolation = 'bicubic',
                    align_corners = True,
                    side = "short",
                    antialias = False
                ), min=0.0, max=1.0 )

            if pred_for_swd.size()!=gt_for_swd.size():
                results_dict['swd'] = None
                return results_dict
                
        else:

            pred_for_swd = pred

        try:
            swd_res = swdobj.run(pred_for_swd.double(),gt_for_swd.double()).item()
        except Exception as e:
            print('Failed SWD > ',str(e))
            print('dimensions', pred_for_swd.shape, gt_for_swd.shape)
            swd_res = None
        
        results_dict['swd'] = swd_res
        
        return results_dict

    def apply(self, predictions: str, gt_path: str) -> Any:
        """
        In this case gt_path will be a glob criteria to the HR images
        """
        
        # These are actually attributes from ds_wrapper
        self.data_path = os.path.dirname(gt_path)
        
        #predictions be like /mlruns/1/6f1b6d86e42d402aa96665e63c44ef91/artifacts'
        guessed_run_id = predictions.split(os.sep)[-3]
        modifier_subfold = [
            k
            for k in self.experiment_info.runs
            if self.experiment_info.runs[k]['run_id']==guessed_run_id
        ][0]
        
        pred_fn_lst = glob(os.path.join(
            os.path.dirname(self.data_path),
            modifier_subfold,
            '*',f'*.{self.ext}' 
        ))

        if len(pred_fn_lst)==0:
            print('Error > empty list "pred_fn_lst" ')
            import pdb; pdb.set_trace()

        if 'LIIF' in pred_fn_lst[0]:
            self._liff_loader_first_time(
                os.path.join(self.data_path,self.img_dir_gt)
                )
        
        stats = { met:0.0 for met in self.metric_names }
        
        fid = piq.FID()
        
        swdobj = SlicedWassersteinDistance(
            n_pyramids           = self.n_pyramids,
            slice_size           = self.slice_size,
            n_descriptors        = self.n_descriptors,
            n_repeat_projection  = self.n_repeat_projection,
            proj_per_repeat      = self.proj_per_repeat,
            device               = self.device,
            return_by_resolution = self.return_by_resolution,
            pyramid_batchsize    = self.pyramid_batchsize
        )

        results_dict_lst = Parallel(n_jobs=self.n_jobs,verbose=10,prefer='threads')(
            delayed(self._parallel)(fid,swdobj,pred_fn)
            for pred_fn in pred_fn_lst
            )
        
        stats = {
            met:np.median([
                r[met]
                for r in results_dict_lst
                if 'float' in type(r[met]).__name__
                ])
            for met in self.metric_names
        }
                
        return stats