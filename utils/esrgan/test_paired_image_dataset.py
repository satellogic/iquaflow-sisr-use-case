import math
import os
import torchvision.utils

from datasets.esrgan import build_dataloader, build_dataset


def test_datasets():
    """Test paired image dataset.
    Args:
        mode: There are three modes: 'lmdb', 'folder', 'meta_info_file'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'DIV2K'
    opt['type'] = 'PairedImageDataset'

    opt['dataroot_gt'] = 'Data/UCMerced_LandUse/train_sub'
    opt['dataroot_lq'] = 'Data/UCMerced_LandUse/train_downx2_sub'
    opt['filename_tmpl'] = '{}'
    opt['io_backend'] = dict(type='disk')

    opt['gt_size'] = 8
    opt['use_flip'] = False
    opt['use_rot'] = False

    opt['use_shuffle'] = False
    opt['num_worker_per_gpu'] = 4
    opt['batch_size_per_gpu'] = 8
    opt['scale'] = 2

    opt['dataset_enlarge_ratio'] = 50

    os.makedirs('tmp', exist_ok=True)

    dataset = build_dataset(opt)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 5:
            break
        print(i)

        lq = data['lq']
        gt = data['gt']
        lq_path = data['lq_path']
        gt_path = data['gt_path']
        print(lq_path, gt_path)
        torchvision.utils.save_image(lq, f'tmp/lq_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        torchvision.utils.save_image(gt, f'tmp/gt_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
