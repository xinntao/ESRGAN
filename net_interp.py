import sys
import torch
from collections import OrderedDict

alpha = float(sys.argv[1])

net_PSNR_path = './models/RRDB_PSNR_x4.pth'
net_ESRGAN_path = './models/RRDB_ESRGAN_x4.pth'
net_interp_path = './models/interp_{:02d}.pth'.format(alpha*10)

net_PSNR = torch.load(net_PSNR_path)
net_ESRGAN = torch.load(net_ESRGAN_path)
net_interp = OrderedDict()

print('Interpolating with alphs = ', alpha)
for k, v_PSNR in net_PSNR.items():
    v_ESRGAN = net_ESRGAN[k]
    net_interp[k] = alpha * v_PSNR + (1 - alpha) * v_ESRGAN

torch.save(net_interp_path, net_interp)
