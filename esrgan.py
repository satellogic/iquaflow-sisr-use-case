import os
import tempfile
import os.path as osp
import cv2
import numpy as np
import torch
import math
import kornia
import PIL.Image as pil_image

from glob import glob
from torchvision import transforms
from models.esrgan import RRDBNet_arch as arch
from lowresgen import LRSimulator
# from custom_iqf import ModelConfS3Loader

def generate_lowres(image_file,scale=3):
    """
    """
    image = pil_image.open(image_file).convert('RGB')
    
    # AFEGIT PER FER EL BLUR
    img_tensor = transforms.ToTensor()(image).unsqueeze_(0)
    sigma = 0.5 * scale
    kernel_size = math.ceil(sigma * 3 + 4)
    kernel_tensor = kornia.filters.get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
    image_blur = kornia.filter2d(img_tensor, kernel_tensor[None])
    image = transforms.ToPILImage()(image_blur.squeeze_(0))
    image = image.resize((int(image.width // scale), int(image.height // scale)), resample=pil_image.BICUBIC)

    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir,'image.tif')
        image.save(fn)
        
        return cv2.imread(fn, cv2.IMREAD_COLOR)

def get_args( zoom ):

    class Args:
        def __init__( self, **kwargs ):

            kwargs.setdefault('images_dir','Data')
            kwargs.setdefault('output_path','output')
            kwargs.setdefault('opt','config/train_esrgan.yaml')
            kwargs.setdefault('model_path','./ESRGAN_1to033_x3_blur/net_g_latest.pth')
            kwargs.setdefault('save_path','output/esrgan')
            kwargs.setdefault('image_file','./Data/test-ds/test/*')
            kwargs.setdefault('auto_resume',True)
            kwargs.setdefault('local_rank',0)
            kwargs.setdefault('launcher',None)
            kwargs.setdefault('zoom',3)

            self.images_dir  = kwargs['images_dir']
            self.output_path = kwargs['output_path']
            self.opt         = kwargs['opt']
            self.model_path  = kwargs['model_path']
            self.save_path   = kwargs['save_path']
            self.image_file  = kwargs['image_file']
            self.auto_resume = kwargs['auto_resume']
            self.local_rank  = kwargs['local_rank']
            self.launcher    = kwargs['launcher']
            self.zoom        = kwargs['zoom']
    
    return Args( zoom = zoom )

# def main():

#     args = get_args( zoom = 3 )

#     model_conf = ModelConfS3Loader(
#         model_fn      = args.model_path,
#         config_fn_lst = [],
#         bucket_name   = "image-quality-framework",
#         algo          = "ESRGAN"
#     )

#     model,_ = model_conf.load_ai_model_and_stuff()

#     print('Model path {:s}. \nTesting...'.format(args.model_path))

#     idx = 0
#     for path in glob( args.image_file ):
#         idx += 1
#         base = osp.splitext(osp.basename(path))[0]
#         print(idx, base)
#         # read images
#         #img = cv2.imread(path, cv2.IMREAD_COLOR)
#         img = generate_lowres( path , scale=args.zoom )

#         img = img * 1.0 / 255
#         img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#         img_LR = img.unsqueeze(0)
#         img_LR = img_LR.to(device)

#         with torch.no_grad():
#             output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#         output = (output * 255.0).round()
#         if not osp.exists(args.save_path):
#             os.makedirs(args.save_path)
#         cv2.imwrite(osp.join(args.save_path, '{:s}.tif'.format(base)), output)

if __name__=="__main__":
    main()