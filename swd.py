
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from PIL import Image
from typing import Any, Dict, Optional, List

class SlicedWassersteinDistance:
    
    def __init__(
        self,
        n_pyramids: Optional[int] = None,
        slice_size: int = 7,
        n_descriptors: int = 128,
        n_repeat_projection: int = 128,
        proj_per_repeat: int = 4,
        device: str = "cpu",
        return_by_resolution: bool = False,
        pyramid_batchsize: int = 128
    ):
        
        self.n_pyramids           = n_pyramids
        self.slice_size           = slice_size
        self.n_descriptors        = n_descriptors
        self.n_repeat_projection  = n_repeat_projection
        self.proj_per_repeat      = proj_per_repeat
        self.device               = device
        self.return_by_resolution = return_by_resolution
        self.pyramid_batchsize    = pyramid_batchsize
        self.device               = device
    
    # Gaussian blur kernel
    def _get_gaussian_kernel(self) -> Any:
        kernel = np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]], np.float32) / 256.0
        gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5)).to(self.device)
        return gaussian_k.double()

    def _pyramid_down(self,image: torch.Tensor) -> torch.Tensor:
        gaussian_k = self._get_gaussian_kernel()        
        # channel-wise conv(important)
        multiband = [F.conv2d(image[:, i:i + 1,:,:], gaussian_k, padding=2, stride=2) for i in range(3)]
        down_image = torch.cat(multiband, dim=1)
        return down_image

    def _pyramid_up(self,image: torch.Tensor) -> torch.Tensor:
        gaussian_k = self._get_gaussian_kernel()
        upsample = F.interpolate(image, scale_factor=2)
        multiband = [F.conv2d(upsample[:, i:i + 1,:,:], gaussian_k, padding=2) for i in range(3)]
        up_image = torch.cat(multiband, dim=1)
        return up_image

    def _gaussian_pyramid(self,original: torch.Tensor) -> List[torch.Tensor]:
        x = original
        # pyramid down
        pyramids = [original]
        for i in range(self.n_pyramids):
            x = self._pyramid_down(x)
            pyramids.append(x)
        return pyramids

    def _laplacian_pyramid(self,original: torch.Tensor) -> List[torch.Tensor]:
        # create gaussian pyramid
        pyramids = self._gaussian_pyramid(original)

        # pyramid up - diff
        laplacian = []
        for i in range(len(pyramids) - 1):
            diff = pyramids[i] - self._pyramid_up(pyramids[i + 1])
            laplacian.append(diff)
        # Add last gaussian pyramid
        laplacian.append(pyramids[len(pyramids) - 1])        
        return laplacian

    def _minibatch_laplacian_pyramid(
        self,
        image: torch.Tensor,
        batch_size : int
    ) -> List[torch.Tensor]:
        n = image.size(0) // batch_size + np.sign(image.size(0) % batch_size)
        pyramids = []
        for i in range(n):
            x = image[i * batch_size:(i + 1) * batch_size]
            p = self._laplacian_pyramid(x.to(self.device))
            p = [x.cpu() for x in p]
            pyramids.append(p)
        del x
        result = []
        for i in range(self.n_pyramids + 1):
            x = []
            for j in range(n):
                x.append(pyramids[j][i])
            result.append(torch.cat(x, dim=0))
        return result

    def _extract_patches(
        self,
        pyramid_layer: torch.Tensor,
        slice_indices: Any,
        unfold_batch_size: int = 128
    ) -> Any:
        assert pyramid_layer.ndim == 4
        n = pyramid_layer.size(0) // unfold_batch_size + np.sign(pyramid_layer.size(0) % unfold_batch_size)
        # random slice 7x7
        p_slice = []
        for i in range(n):
            # [unfold_batch_size, ch, n_slices, slice_size, slice_size]
            ind_start = i * unfold_batch_size
            ind_end = min((i + 1) * unfold_batch_size, pyramid_layer.size(0))
            x = pyramid_layer[ind_start:ind_end].unfold(
                    2, self.slice_size, 1).unfold(3, self.slice_size, 1).reshape(
                    ind_end - ind_start, pyramid_layer.size(1), -1, self.slice_size, self.slice_size)
            # [unfold_batch_size, ch, n_descriptors, slice_size, slice_size]
            x = x[:,:, slice_indices,:,:]
            # [unfold_batch_size, n_descriptors, ch, slice_size, slice_size]
            p_slice.append(x.permute([0, 2, 1, 3, 4]))
        # sliced tensor per layer [batch, n_descriptors, ch, slice_size, slice_size]
        x = torch.cat(p_slice, dim=0)
        # normalize along ch
        std, mean = torch.std_mean(x, dim=(0, 1, 3, 4), keepdim=True)
        x = (x - mean) / (std + 1e-8)
        # reshape to 2rank
        x = x.reshape(-1, 3 * self.slice_size * self.slice_size)
        return x.double()

    def run(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> torch.Tensor:
        
        # n_repeat_projectton * proj_per_repeat = 512
        # Please change these values according to memory usage.
        # original = n_repeat_projection=4, proj_per_repeat=128    
        assert image1.size() == image2.size()
        assert image1.ndim == 4 and image2.ndim == 4

        if self.n_pyramids is None:
            
            self.n_pyramids = int(np.rint(np.log2(image1.size(2) // 16)))
            
        with torch.no_grad():
            # minibatch laplacian pyramid for cuda memory reasons
            
            pyramid1 = self._minibatch_laplacian_pyramid(image1,self.pyramid_batchsize)
            pyramid2 = self._minibatch_laplacian_pyramid(image2,self.pyramid_batchsize)
            
            result = []

            for i_pyramid in range(self.n_pyramids + 1):
                # indices
                n = (pyramid1[i_pyramid].size(2) - 6) * (pyramid1[i_pyramid].size(3) - 6)
                indices = torch.randperm(n)[:self.n_descriptors]

                # extract patches on CPU
                # patch : 2rank (n_image*n_descriptors, slice_size**2*ch)
                p1 = self._extract_patches(pyramid1[i_pyramid], indices, unfold_batch_size=128)
                p2 = self._extract_patches(pyramid2[i_pyramid], indices, unfold_batch_size=128)

                p1, p2 = p1.to(self.device), p2.to(self.device)

                distances = []
                for j in range(self.n_repeat_projection):
                    # random
                    rand = torch.randn(p1.size(1), self.proj_per_repeat).to(self.device)  # (slice_size**2*ch)
                    rand = ( rand / torch.std(rand, dim=0, keepdim=True) ).double()  # noramlize
                    # projection
                    proj1 = torch.matmul(p1, rand)
                    proj2 = torch.matmul(p2, rand)
                    proj1, _ = torch.sort(proj1, dim=0)
                    proj2, _ = torch.sort(proj2, dim=0)
                    d = torch.abs(proj1 - proj2)
                    distances.append(torch.mean(d))

                # swd
                result.append(torch.mean(torch.stack(distances)))

            # average over resolution
            result = torch.stack(result) * 1e3
            if self.return_by_resolution:
                return result.cpu()
            else:
                return torch.mean(result).cpu()

if __name__=='__main__':
    
    torch.manual_seed(123) # fix seed
    x1 = torch.rand(2, 3, 256, 256).double()
    x2 = torch.rand(2, 3, 256, 256).double()
    swdobj = SlicedWassersteinDistance()
    out = swdobj.run(x1,x2)
    print(out.item())