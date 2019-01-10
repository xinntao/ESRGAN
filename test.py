import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch
from pathlib import Path

model_path = sys.argv[1]  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR/*'

model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)
model_used = Path('{:s}'.format(model_path)).stem # get model name to simple string for filename use

print('Model path {:s}. \nTesting...'.format(model_path))


amend = 2 #File already exists, lets amend a 2 to the end of the file name
idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = os.path.splitext(os.path.basename(path))[0]
    print(idx, base)
    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    while os.path.exists('results/{:s}_%s_%s.png'.format(base) % (model_used, amend)):
        amend += 1

    if os.path.exists('results/{:s}_%s.png'.format(base) % model_used):
        cv2.imwrite('results/{:s}_%s_%s.png'.format(base) % (model_used, amend), output)

    else: 
        cv2.imwrite('results/{:s}_%s.png'.format(base) % model_used, output)

