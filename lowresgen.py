import torch

import numpy as np

import torch.nn.functional as F

from typing import List, Optional, Tuple, Union

class LRSimulator(object):
    
    """ This class degradates an image and generates its lower res """
    
    def __init__(self,img_array,zoom):
        
        self.img_array = img_array
        self.scale = 1/zoom

    def add_blur(self) -> "np.ndarray":
        
        img = self._image_to_tensor(self.img_array.copy()).float()[None]
        
        sigma = 0.5*(1/self.scale)
        kernel_size = int(sigma*3 + 4)
        if kernel_size%2==0:
            kernel_size=+1

        kernel_tensor = self._get_gaussian_kernel2d((kernel_size,kernel_size), (sigma, sigma))
        
        blurred = self._filter2d(img, kernel_tensor[None])

        return blurred

    def resize_to_lr(self,input_img) -> "np.ndarray":

        blurred_resize = self._rescale(input_img, float(self.scale), 'bicubic')
        blurred_resize = blurred_resize.numpy()[0].transpose(1,2,0).astype(np.uint8)
        return blurred_resize

    def generate_low_resolution_image(self) -> "np.ndarray":
        
        input_img = self.add_blur()
        blurred_resize = self.resize_to_lr(input_img)
        return blurred_resize
    
    def _compute_padding(self,kernel_size: List[int]) -> List[int]:
        """Computes padding tuple."""
        # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
        assert len(kernel_size) >= 2, kernel_size
        computed = [k // 2 for k in kernel_size]

        # for even kernels we need to do asymetric padding :(

        out_padding = 2 * len(kernel_size) * [0]

        for i in range(len(kernel_size)):
            computed_tmp = computed[-(i + 1)]
            if kernel_size[i] % 2 == 0:
                padding = computed_tmp - 1
            else:
                padding = computed_tmp
            out_padding[2 * i + 0] = padding
            out_padding[2 * i + 1] = computed_tmp
        return out_padding

    def _gaussian(self,window_size: int, sigma: float) -> torch.Tensor:
        device, dtype = None, None
        if isinstance(sigma, torch.Tensor):
            device, dtype = sigma.device, sigma.dtype
        x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        if window_size % 2 == 0:
            x = x + 0.5
        gauss = torch.exp((-x.pow(2.0) / (2 * sigma ** 2)))
        return gauss / gauss.sum()
    
    def _side_to_image_size(self,side_size: int, aspect_ratio: float, side: str = "short") -> Tuple[int, int]:
        if side not in ("short", "long", "vert", "horz"):
            raise ValueError(f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'")
        if side == "vert":
            return side_size, int(side_size * aspect_ratio)
        if side == "horz":
            return int(side_size / aspect_ratio), side_size
        if (side == "short") ^ (aspect_ratio < 1.0):
            return side_size, int(side_size * aspect_ratio)
        return int(side_size / aspect_ratio), side_size
    
    def _resize(
        self,
        in_tensor: torch.Tensor,
        size: Union[int, Tuple[int, int]],
        interpolation: str = 'bilinear',
        align_corners: Optional[bool] = None,
        side: str = "short",
        antialias: bool = False,
    ) -> torch.Tensor:
        r"""Resize the input torch.Tensor to the given size.
        .. image:: _static/img/resize.png
        Args:
            tensor: The image tensor to be skewed with shape of :math:`(..., H, W)`.
                `...` means there can be any number of dimensions.
            size: Desired output size. If size is a sequence like (h, w),
                output size will be matched to this. If size is an int, smaller edge of the image will
                be matched to this number. i.e, if height > width, then image will be rescaled
                to (size * height / width, size)
            interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` |
                'bicubic' | 'trilinear' | 'area'.
            align_corners: interpolation flag.
            side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
                or ``'horz'``.
            antialias: if True, then image will be filtered with Gaussian before downscaling.
                No effect for upscaling.
        Returns:
            The resized tensor with the shape as the specified size.
        Example:
            >>> img = torch.rand(1, 3, 4, 4)
            >>> out = resize(img, (6, 8))
            >>> print(out.shape)
            torch.Size([1, 3, 6, 8])
        """
        if not isinstance(in_tensor, torch.Tensor):
            raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(in_tensor)))

        if len(in_tensor.shape) < 2:
            raise ValueError('Input tensor must have at least two dimensions. Got {}'.format(len(in_tensor.shape)))

        input_size = h, w = in_tensor.shape[-2:]
        if isinstance(size, int):
            aspect_ratio = w / h
            size = self._side_to_image_size(size, aspect_ratio, side)

        if size == input_size:
            return in_tensor

        factors = (h / size[0], w / size[1])

        # We do bluring only for downscaling
        antialias = antialias and (max(factors) > 1)

        if antialias:
            # First, we have to determine sigma
            sigmas = (max(factors[0], 1.0), max(factors[1], 1.0))

            # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
            # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
            # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
            ks = int(2.0 * 2 * sigmas[0] + 1), int(2.0 * 2 * sigmas[1] + 1)
            in_tensor = kornia.filters.gaussian_blur2d(in_tensor, ks, sigmas)

        output = torch.nn.functional.interpolate(in_tensor, size=size, mode=interpolation, align_corners=align_corners)
        return output

    def _get_gaussian_kernel1d(self,kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:

        if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
            raise TypeError("kernel_size must be an odd positive integer. " "Got {}".format(kernel_size))
        window_1d: torch.Tensor = self._gaussian(kernel_size, sigma)
        return window_1d
    
    def _image_to_tensor(self,image: "np.ndarray", keepdim: bool = True) -> torch.Tensor:
        """Convert a numpy image to a PyTorch 4d tensor image.
        Args:
            image: image of the form :math:`(H, W, C)`, :math:`(H, W)` or
                :math:`(B, H, W, C)`.
            keepdim: If ``False`` unsqueeze the input image to match the shape
                :math:`(B, H, W, C)`.
        Returns:
            tensor of the form :math:`(B, C, H, W)` if keepdim is ``False``,
                :math:`(C, H, W)` otherwise.
        """
        if len(image.shape) > 4 or len(image.shape) < 2:
            raise ValueError("Input size must be a two, three or four dimensional array")

        input_shape = image.shape
        tensor: torch.Tensor = torch.from_numpy(image)

        if len(input_shape) == 2:
            # (H, W) -> (1, H, W)
            tensor = tensor.unsqueeze(0)
        elif len(input_shape) == 3:
            # (H, W, C) -> (C, H, W)
            tensor = tensor.permute(2, 0, 1)
        elif len(input_shape) == 4:
            # (B, H, W, C) -> (B, C, H, W)
            tensor = tensor.permute(0, 3, 1, 2)
            keepdim = True  # no need to unsqueeze
        else:
            raise ValueError("Cannot process image with shape {}".format(input_shape))

        return tensor.unsqueeze(0) if not keepdim else tensor
    
    def _rescale(
        self,
        in_tensor: torch.Tensor,
        factor: Union[float, Tuple[float, float]],
        interpolation: str = "bilinear",
        align_corners: Optional[bool] = None,
        antialias: bool = False,
    ) -> torch.Tensor:
        r"""Rescale the input torch.Tensor with the given factor.
        .. image:: _static/img/rescale.png
        Args:
            in_tensor: The image tensor to be scale with shape of :math:`(B, C, H, W)`.
            factor: Desired scaling factor in each direction. If scalar, the value is used
                for both the x- and y-direction.
            interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` |
                ``'bicubic'`` | ``'trilinear'`` | ``'area'``.
            align_corners: interpolation flag.
            side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``,
                or ``'horz'``.
            antialias: if True, then image will be filtered with Gaussian before downscaling.
                No effect for upscaling.
        Returns:
            The rescaled tensor with the shape as the specified size.
        Example:
            >>> img = torch.rand(1, 3, 4, 4)
            >>> out = rescale(img, (2, 3))
            >>> print(out.shape)
            torch.Size([1, 3, 8, 12])
        """
        if isinstance(factor, float):
            factor_vert = factor_horz = factor
        else:
            factor_vert, factor_horz = factor

        height, width = in_tensor.size()[-2:]
        size = (int(height * factor_vert), int(width * factor_horz))
        return self._resize(in_tensor, size, interpolation=interpolation, align_corners=align_corners, antialias=antialias)
    
    def _get_gaussian_kernel1d(self,kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:
        r"""Function that returns Gaussian filter coefficients.
        Args:
            kernel_size (int): filter size. It should be odd and positive.
            sigma (float): gaussian standard deviation.
            force_even (bool): overrides requirement for odd kernel size.
        Returns:
            Tensor: 1D tensor with gaussian filter coefficients.
        Shape:
            - Output: :math:`(\text{kernel_size})`
        Examples:
            >>> get_gaussian_kernel1d(3, 2.5)
            tensor([0.3243, 0.3513, 0.3243])
            >>> get_gaussian_kernel1d(5, 1.5)
            tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
        """
        if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
            raise TypeError("kernel_size must be an odd positive integer. " "Got {}".format(kernel_size))
        window_1d: torch.Tensor = self._gaussian(kernel_size, sigma)
        return window_1d
    
    def _get_gaussian_erf_kernel1d(self, kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:
        r"""Function that returns Gaussian filter coefficients by interpolating the error fucntion,
        adapted from:
        https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
        Args:
            kernel_size: filter size. It should be odd and positive.
            sigma: gaussian standard deviation.
            force_even: overrides requirement for odd kernel size.
        Returns:
            1D tensor with gaussian filter coefficients.
        Shape:
            - Output: :math:`(\text{kernel_size})`
        Examples:
            >>> get_gaussian_erf_kernel1d(3, 2.5)
            tensor([0.3245, 0.3511, 0.3245])
            >>> get_gaussian_erf_kernel1d(5, 1.5)
            tensor([0.1226, 0.2331, 0.2887, 0.2331, 0.1226])
        """
        if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
            raise TypeError("kernel_size must be an odd positive integer. " "Got {}".format(kernel_size))
        window_1d = gaussian_discrete_erf(kernel_size, sigma)
        return window_1d


    def _get_gaussian_kernel2d(self,
        kernel_size: Tuple[int, int], sigma: Tuple[float, float], force_even: bool = False
    ) -> torch.Tensor:
        r"""Function that returns Gaussian filter matrix coefficients.
        Args:
            kernel_size: filter sizes in the x and y direction.
             Sizes should be odd and positive.
            sigma: gaussian standard deviation in the x and y
             direction.
            force_even: overrides requirement for odd kernel size.
        Returns:
            2D tensor with gaussian filter matrix coefficients.
        Shape:
            - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`
        Examples:
            >>> get_gaussian_kernel2d((3, 3), (1.5, 1.5))
            tensor([[0.0947, 0.1183, 0.0947],
                    [0.1183, 0.1478, 0.1183],
                    [0.0947, 0.1183, 0.0947]])
            >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
            tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                    [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                    [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
        """
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise TypeError("kernel_size must be a tuple of length two. Got {}".format(kernel_size))
        if not isinstance(sigma, tuple) or len(sigma) != 2:
            raise TypeError("sigma must be a tuple of length two. Got {}".format(sigma))
        ksize_x, ksize_y = kernel_size
        sigma_x, sigma_y = sigma
        kernel_x: torch.Tensor = self._get_gaussian_kernel1d( ksize_x, sigma_x, force_even )
        kernel_y: torch.Tensor = self._get_gaussian_kernel1d( ksize_y, sigma_y, force_even )
        kernel_2d: torch.Tensor = torch.matmul( kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t() )
        return kernel_2d

    def _filter2d( self,
        in_tensor: torch.Tensor, kernel: torch.Tensor, border_type: str = 'reflect', normalized: bool = False
    ) -> torch.Tensor:
        r"""Convolve a tensor with a 2d kernel.
        The function applies a given kernel to a tensor. The kernel is applied
        independently at each depth channel of the tensor. Before applying the
        kernel, the function applies padding according to the specified mode so
        that the output remains in the same shape.
        Args:
            in_tensor: the input tensor with shape of
              :math:`(B, C, H, W)`.
            kernel: the kernel to be convolved with the input
              tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
            border_type: the padding mode to be applied before convolving.
              The expected modes are: ``'constant'``, ``'reflect'``,
              ``'replicate'`` or ``'circular'``.
            normalized: If True, kernel will be L1 normalized.
        Return:
            torch.Tensor: the convolved tensor of same size and numbers of channels
            as the input with shape :math:`(B, C, H, W)`.
        Example:
            >>> in_tensor = torch.tensor([[[
            ...    [0., 0., 0., 0., 0.],
            ...    [0., 0., 0., 0., 0.],
            ...    [0., 0., 5., 0., 0.],
            ...    [0., 0., 0., 0., 0.],
            ...    [0., 0., 0., 0., 0.],]]])
            >>> kernel = torch.ones(1, 3, 3)
            >>> filter2d(in_tensor, kernel)
            tensor([[[[0., 0., 0., 0., 0.],
                      [0., 5., 5., 5., 0.],
                      [0., 5., 5., 5., 0.],
                      [0., 5., 5., 5., 0.],
                      [0., 0., 0., 0., 0.]]]])
        """
        if not isinstance(in_tensor, torch.Tensor):
            raise TypeError("Input border_type is not torch.Tensor. Got {}".format(type(in_tensor)))

        if not isinstance(kernel, torch.Tensor):
            raise TypeError("Input border_type is not torch.Tensor. Got {}".format(type(kernel)))

        if not isinstance(border_type, str):
            raise TypeError("Input border_type is not string. Got {}".format(type(kernel)))

        if not len(in_tensor.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(in_tensor.shape))

        if not len(kernel.shape) == 3 and kernel.shape[0] != 1:
            raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}".format(kernel.shape))

        # prepare kernel
        b, c, h, w = in_tensor.shape
        tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(in_tensor)

        if normalized:
            tmp_kernel = normalize_kernel2d(tmp_kernel)

        tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

        # pad the input tensor
        height, width = tmp_kernel.shape[-2:]
        padding_shape: List[int] = self._compute_padding([height, width])
        input_pad: torch.Tensor = F.pad(in_tensor, padding_shape, mode=border_type)

        # kernel and input tensor reshape to align element-wise or batch-wise params
        tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
        input_pad = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-2), input_pad.size(-1))

        # convolve the tensor with the kernel.
        output = F.conv2d(input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

        return output.view(b, c, h, w)

if __name__ == "__main__":
    
    
    
    def generate_low_resolution_image(img_array, scale=2):
        
        import kornia
        
        img = kornia.image_to_tensor(img_array.copy()).float()[None]

        sigma = 0.5*scale
        kernel_size = int(sigma*3 + 4)
        if kernel_size%2==0:
            kernel_size=+1

        kernel_tensor = kornia.filters.get_gaussian_kernel2d((kernel_size,kernel_size), (sigma, sigma))
        blurred = kornia.filter2d(img, kernel_tensor[None])
        blurred_resize = kornia.geometry.rescale(blurred, 1/scale, 'bicubic')
        print(blurred_resize.size())
        blurred_resize = blurred_resize.numpy()[0].transpose(1,2,0).astype(np.uint8)
        return blurred_resize
    
    scale = 2
    
    hr = np.random.rand(256,256,3)*255
    
    lr1 = generate_low_resolution_image(hr, scale=scale)
    
    lrs = LRSimulator(
        hr, scale
    )
    
    lr = lrs.generate_low_resolution_image()
    
    print( 'hr', hr.shape , 'lr' , lr.shape , 'lr1' , lr1.shape )
    
    assert lr.shape == lr1.shape , 'output shape is not the expected'