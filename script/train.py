# The Code written by Ali Babolhaveji @ 6/1/2020


import sys
import yaml
import os
import gc
import pandas as pd

import torch
import torch.nn as nn
package_path = '..'
# package_path = '../../script2/Result/nuScenes/exp3/sources/'
if not package_path in sys.path:
    sys.path.append(package_path)

from lib import  ClogLossDataset_from_compressed_data
from lib import my_deep_clag_loss

from torch.utils.data import DataLoader
from lib import  train_val
from lib import create_new_experiment
import pickle

from tqdm import tqdm

config = './config.yml'
with open (config , 'rb') as f:
    config = yaml.load(f ,Loader=yaml.FullLoader)

# get a backup from important file in the project
this_expr = create_new_experiment(config)
this_expr.copyanything(os.path.basename(__file__) ,os.path.join(this_expr.sourefiles_path,'script') )
this_expr.copyanything(os.path.basename('./config.yml') ,os.path.join(this_expr.sourefiles_path,'script') )
this_expr.copyanything(os.path.abspath('../lib'), os.path.join(this_expr.sourefiles_path, 'lib'))

# create train and validation dataset
fold = 0
train_dataset = ClogLossDataset_from_compressed_data(config , split='train' ,fold = fold )
val_dataset   = ClogLossDataset_from_compressed_data(config , split='val' ,fold = fold)



this_expr.print_log(f"Trin dataset: {len(train_dataset)} objects")
this_expr.print_log(f"Val dataset : {len(val_dataset)} objects")


# for data in tqdm(train_dataset):
#     pass
# for data in tqdm(val_dataset):
#     pass



device = torch.device(f"{config['train']['device']}" if torch.cuda.is_available() else "cpu")

this_expr.print_log(f"Device      : {device}")

train_loader = DataLoader(train_dataset, batch_size=config['train']['trainBatchSize'],
                          num_workers=config['train']['num_Workers'], shuffle=True , pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=config['train']['valBatchSize'], num_workers=config['train']['num_Workers'],
                        shuffle=True ,pin_memory=True)
                        
                        
                        

vid_tensor , meta = next(iter(train_loader))

model = my_deep_clag_loss()
model.to(device)

if config['model']['load']['flag']:
    state_dict = torch.load(config['model']['load']['path'])
    model.load_state_dict(state_dict)
    this_expr.print_log(f"[info] Model is loaded. [from {config['model']['load']['path']}]")

    
    
    
# inputs = torch.randn((2, 3, 200, 150, 150)).to(device)

vid_tensor = vid_tensor.to(device)
print(model(vid_tensor).shape)


# optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr']['init'], weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=config['train']['lr']['init'])
optimizer = torch.optim.SGD(model.parameters(), lr=config['train']['lr']['init'], momentum=0.9)

# criterion = nn.BCELoss()
criterion = nn.MSELoss()

best_loss = 1e+10
history = pd.DataFrame()

for epoch in range(config['train']['startEpoch'], config['train']['endEpoch']):  # loop over the dataset multiple times
    this_expr.print_log(f"=======================  Epoch {epoch} / {config['train']['endEpoch']}  =======================")
    gc.collect()


    _, history             =   train_val(model, 
                                         train_loader, 
                                         epoch, device, 
                                         optimizer, 
                                         criterion, 
                                         config, 
                                         mode='train',
                                         history =history,
                                         this_expr = this_expr )
    
    
    curr_val_loss, history =   train_val(model, 
                                         val_loader, 
                                         epoch, device, 
                                         optimizer, 
                                         criterion, 
                                         config, 
                                         mode='val',
                                         history =history,
                                         this_expr = this_expr)
                                         
    with open(os.path.join(this_expr.model_checkpoints_path, f'history.pkl'), 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    this_expr.print_log('current_val_loss: ', f"{curr_val_loss}")
    
    if curr_val_loss < best_loss:
        model_save_format = f"model_E{epoch:03d}_Loss{curr_val_loss:.6f}.pt"
        # torch.save(model.state_dict(), os.path.join(experiment_name ,f"./model_E{epoch:03d}_Loss{curr_val_loss:.6f}.pt"))
        torch.save(model.state_dict(), os.path.join(this_expr.model_checkpoints_path, model_save_format))
        # print (f"./model_E{epoch:03d}_Loss{curr_val_loss:.6f}.pt  is saved.")
        this_expr.print_log(os.path.join(this_expr.model_checkpoints_path, model_save_format) + " is saved.")

        best_loss = curr_val_loss
        with open(os.path.join(this_expr.model_checkpoints_path, f'history_best_model.pkl'), 'wb') as handle:
            pickle.dump(history, handle, protocol = pickle.HIGHEST_PROTOCOL)
