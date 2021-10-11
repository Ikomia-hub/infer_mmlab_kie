import numpy as np

kie_models = {
    'SDMGR': {
        'config': 'sdmgr/sdmgr_unet16_60e_wildreceipt.py',
        'ckpt':
            'sdmgr/sdmgr_unet16_60e_wildreceipt_20210520-7489e6de.pth'
    }
}


def polygon2bbox(pts):
    x = np.min(pts[::2])
    y = np.min(pts[1::2])
    w = np.max(pts[::2]) - x
    h = np.max(pts[1::2]) - y
    return [int(x), int(y), int(w), int(h)]

def get_classes(dict_file):

    with open(dict_file,"r") as f:
        for line in f.readlines():
            print(line)
