# The Code written by Ali Babolhaveji @ 6/1/2020


import sys
import torch
import yaml
from torch.utils.data import DataLoader

package_path = '..'
# package_path = '../../script2/Result/nuScenes/exp3/sources/'
if not package_path in sys.path:
    sys.path.append(package_path)


from lib import my_deep_clag_loss
from lib import ClogLossDataset

config = './config.yml'
with open (config , 'rb') as f:
    config = yaml.load(f ,Loader=yaml.FullLoader)

train_Dataset = ClogLossDataset(config)


train_loader = DataLoader(train_Dataset, batch_size=2,
                          num_workers=0, shuffle=True)

inputs = next(iter(train_loader))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = my_deep_clag_loss()
model.to(device)
# inputs = torch.randn((2, 3, 200, 150, 150)).to(device)

inputs = inputs.to(device)
print(model(inputs).shape)