import torch, torchvision.models as models, netvlad
from os.path import join

run_dir = r'.\runs\Nov25_14-46-48_vgg16_netvlad'
ckpt_path = join(run_dir, 'checkpoints', 'model_best.pth.tar')

ckpt = torch.load(ckpt_path, map_location='cpu')   # or map_location='cuda' if you have GPU
state = ckpt['state_dict']

# Rebuild model architecture exactly like main.py (example for vgg16)
encoder = models.vgg16(pretrained=False).features[:-2]   # mimic how encoder was built
encoder = torch.nn.Sequential(*list(encoder))
# wrap encoder into a model container as in main.py
model = torch.nn.Module()
model.add_module('encoder', encoder)
# create NetVLAD with the same params used in training:
net_vlad = netvlad.NetVLAD(num_clusters=32, dim=512, vladv2=False)
model.add_module('pool', net_vlad)

# load weights
model.load_state_dict(state)
model.eval()

# now use model.encoder and model.pool to extract features
# e.g. x = preprocess(image_tensor); desc = model.pool(model.encoder(x.unsqueeze(0)))