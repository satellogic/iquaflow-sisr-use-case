import os
import shutil
import cv2

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image as pil_image

#########################
# Visual comparison
#########################

def scatter_plots(df):
    
    for pair_metrics in [
        ['ssim','psnr'],
        ['fid','swd'],
        ['rer_0','snr']
    ]:

        met1, met2 = pair_metrics

        fig, ax = plt.subplots()

        marker_lst = []

        for i in df.index:

            if not '#' in df['ds_modifier'][i]:
                marker = 'X'
            else:
                marker = 'o'

            ax.scatter(
                (
                    df[met1][i]
                    if met1!='rer_0'
                    else np.nanmean([df['rer_0'][i],df['rer_1'][i],df['rer_2'][i]])
                ),
                df[met2][i],
                s=250.,
                marker=marker,
                label=df['ds_modifier'][i],
                alpha=0.5,
                edgecolors='none'
            )

        ax.set_xlabel(('rer' if met1=='rer_0' else met1))
        ax.set_ylabel(met2)
        ax.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)

        plt.show()

def visual_comp():
    
    lst_lst = [
        glob(fr"./Data/test-ds/test/*"),
        glob(fr"./Data/test-ds#sisr+MSRN_MSRN_nonoise-MSRN_1to033-model_epoch_1500/test/*"),
        glob(fr"./Data/test-ds#sisr+ESRGAN_ESRGAN_1to033_x3_blur-net_g_latest/test/*"),
        glob(fr"./Data/test-ds#sisr+FSRCNN_test_scale3_FSRCNN_1to033_x3_blur-best/test/*"),
        glob(fr"./Data/test-ds#sisr+LIIF_LIIF_config_test_liif_LIIF_blur-epoch-best/test/*")
    ]

    print('\t\t"GT"\t\t"MSRN"\t\t"ESRGAN"\t\t"FSRCNN"\t\t"LIIF"')

    for enu,fn in enumerate(lst_lst[0]):

        if enu>20:
            break

        n_alg = len(lst_lst)

        arr_lst = [
            # cv2.imread( [ 
            #     f for f in lst_lst[i]
            #     if os.path.basename(f)==os.path.basename(fn)
            # ][0])
            # if i<2 else 
            cv2.imread( [ 
                f for f in lst_lst[i]
                if os.path.basename(f)==os.path.basename(fn)
            ][0] )[...,::-1]
            for i in range( n_alg ) 
        ]

        fig,ax = plt.subplots(1, n_alg ,figsize=(20,7), gridspec_kw={'wspace':0, 'hspace':0},squeeze=True)
        for i in range( n_alg ):
            ax[i].imshow( arr_lst[i])
            ax[i].axis('off')
        plt.show()

        fig,ax = plt.subplots(1, n_alg ,figsize=(20,7), gridspec_kw={'wspace':0, 'hspace':0},squeeze=True)
        for i in range( n_alg ):
            ax[i].imshow( arr_lst[i][75:-75:,75:-75:,:])
            ax[i].axis('off')
        plt.show()