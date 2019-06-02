import os
import torch
import RRDBNet_arch as arch

pretrained_net = torch.load('./models/RRDB_ESRGAN_x4.pth')
save_path = './models/RRDB_ESRGAN_x4.pth'

crt_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
crt_net = crt_model.state_dict()

load_net_clean = {}
for k, v in pretrained_net.items():
    if k.startswith('module.'):
        load_net_clean[k[7:]] = v
    else:
        load_net_clean[k] = v
pretrained_net = load_net_clean

print('###################################\n')
tbd = []
for k, v in crt_net.items():
    tbd.append(k)

# directly copy
for k, v in crt_net.items():
    if k in pretrained_net and pretrained_net[k].size() == v.size():
        crt_net[k] = pretrained_net[k]
        tbd.remove(k)

crt_net['conv_first.weight'] = pretrained_net['model.0.weight']
crt_net['conv_first.bias'] = pretrained_net['model.0.bias']

for k in tbd.copy():
    if 'RDB' in k:
        ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
        if '.weight' in k:
            ori_k = ori_k.replace('.weight', '.0.weight')
        elif '.bias' in k:
            ori_k = ori_k.replace('.bias', '.0.bias')
        crt_net[k] = pretrained_net[ori_k]
        tbd.remove(k)

crt_net['trunk_conv.weight'] = pretrained_net['model.1.sub.23.weight']
crt_net['trunk_conv.bias'] = pretrained_net['model.1.sub.23.bias']
crt_net['upconv1.weight'] = pretrained_net['model.3.weight']
crt_net['upconv1.bias'] = pretrained_net['model.3.bias']
crt_net['upconv2.weight'] = pretrained_net['model.6.weight']
crt_net['upconv2.bias'] = pretrained_net['model.6.bias']
crt_net['HRconv.weight'] = pretrained_net['model.8.weight']
crt_net['HRconv.bias'] = pretrained_net['model.8.bias']
crt_net['conv_last.weight'] = pretrained_net['model.10.weight']
crt_net['conv_last.bias'] = pretrained_net['model.10.bias']

torch.save(crt_net, save_path)
print('Saving to ', save_path)
