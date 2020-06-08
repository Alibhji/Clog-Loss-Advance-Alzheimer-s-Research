# The Code written by Ali Babolhaveji @ 6/1/2020


import sys
import torch
import yaml
import os

package_path = '..'
# package_path = '../../script2/Result/nuScenes/exp3/sources/'
if not package_path in sys.path:
    sys.path.append(package_path)

from lib import  ClogLossDataset_from_compressed_data
from lib import my_deep_clag_loss

from torch.utils.data import DataLoader
from lib import  train_val
from lib import create_new_experiment

config = './config.yml'
with open (config , 'rb') as f:
    config = yaml.load(f ,Loader=yaml.FullLoader)

this_expr = create_new_experiment(config)
this_expr.copyanything(os.path.basename(__file__) ,os.path.join(this_expr.sourefiles_path,'script') )
this_expr.copyanything(os.path.basename('./config.yml') ,os.path.join(this_expr.sourefiles_path,'script') )
this_expr.copyanything(os.path.abspath('../lib'), os.path.join(this_expr.sourefiles_path, 'lib'))

# create train and validation dataset
train_Dataset = ClogLossDataset_from_compressed_data(config)


train_loader = DataLoader(train_Dataset, batch_size=2,
                          num_workers=0, shuffle=True)

vid_tensor , meta = next(iter(train_loader))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = my_deep_clag_loss()
model.to(device)
# inputs = torch.randn((2, 3, 200, 150, 150)).to(device)

vid_tensor = vid_tensor.to(device)
print(model(vid_tensor).shape)