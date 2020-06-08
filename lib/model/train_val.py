from tqdm import tqdm
import time
import torch
import copy


def loss_function (output ,target,criterion ):
    rows = range(len(target))
    oneHotLabel = torch.zeros(len(target),2)
    oneHotLabel[rows ,target.numpy()==1] =1
    oneHotLabel = oneHotLabel.to(output.device)

    return criterion(output , oneHotLabel)
    
    
def lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7 ,logger=None):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch or epoch==0:
        return optimizer

    if logger is not None:
        cur_lr = get_lr(optimizer)

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay

    if logger is not None:
        logger.info(f"[{os.path.basename(__file__)}] " + f"'lr' is updated from [{cur_lr} to {get_lr(optimizer)}]")

    return optimizer
 
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        lr_= param_group['lr']
    return lr_


def train_val(model, data_loader, epoch, device, optimizer, criterion, config, mode='train',logger=None , history=None ,this_expr =None):
    
    def Logger(msg, logger=logger):
        if logger is not None:
            logger.info(f"[{os.path.basename(__file__)}] "+msg)
        else:
            pass
            
    cfg = config
    ep_since = time.time()

    running_loss = 0.0
    running_BEV = 0.0
    running_pts_3D = 0.0
    r = tqdm(data_loader)



    if mode == 'train':
        model.train()




        optimizer = lr_scheduler(optimizer, epoch,
                                     lr_decay=cfg['train']['lr']['lr_decay'],
                                     lr_decay_epoch=cfg['train']['lr']['lr_decay_epoch'] ,
                                     logger=logger)

        lr__ = get_lr(optimizer)


    elif mode == 'val':
        model.eval()
        
    #this_expr.print_log(f" Mode [mode] is set to: {mode}" , print_=False)


    for i, data in enumerate(r, 0):
        torch.cuda.empty_cache()
        since = time.time()

        # # get the inputs; data is a list of [inputs, labels]
        img_tensor , meta = data
        
        img_tensor = img_tensor.to(device)


        # # zero the parameter gradients
        if mode == 'train':

            output = model(img_tensor)

            loss = loss_function(output ,meta['stalled'] ,criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = copy.deepcopy(loss.cpu().detach().numpy())
            running_loss += loss.item() / len(data_loader)

            time_elapsed_batch = time.time() - since
            time_elapsed = time.time() - ep_since
            dis_msg =\
                f"[T] loss: {loss_value:2.5f} total_loss: {running_loss:02.6f} " \
                      f"[lr: {lr__:0.07f}] [Time {time_elapsed_batch: 03.3f} {time_elapsed: 03.3f}]"
            r.set_description(dis_msg)
            this_expr.writer.add_scalar('training_batch_loss',
                              loss_value,
                              epoch * len(data_loader) + i)

            if history is not None:
                history.loc[epoch * len(data_loader) + i, 'train_loss_value'] = loss_value
                history.loc[epoch * len(data_loader) + i, 'train_running_loss'] = running_loss


            if(i==4):
                break

        else:

            output = model(img_tensor)
            
            loss = loss_function(output ,meta['stalled'] ,criterion)


            loss_value = copy.deepcopy(loss.cpu().detach().numpy())
            running_loss += loss.item() / len(data_loader)

            time_elapsed_batch = time.time() - since
            time_elapsed = time.time() - ep_since
            dis_msg =\
                f"[E] loss: {loss_value:2.5f} total_loss: {running_loss:02.6f} " \
                      f"[Time {time_elapsed_batch: 03.3f} {time_elapsed: 03.3f}]"
            r.set_description(dis_msg)

            this_expr.writer.add_scalar('eval_batch_loss',
                              loss_value,
                              epoch * len(data_loader) + i)

            if history is not None:
                history.loc[epoch * len(data_loader) + i, 'eval_loss_value'] = loss_value
                history.loc[epoch * len(data_loader) + i, 'eval_running_loss'] = running_loss
            
            if(i==4):
                break
            

    if mode == 'train':
        this_expr.print_log(dis_msg , print_=False)

        if history is not None:
            history.loc[epoch, 'train_total_loss'] = running_loss
            history.loc[epoch, 'lr'] = lr__
            history.loc[epoch, 'train_time'] = time.time() - ep_since



        this_expr.writer.add_scalar('training_total_loss',
                      loss_value,
                      epoch * len(data_loader) + i)
        this_expr.writer.add_scalar('lr',
                          lr__,
                          epoch * len(data_loader) + i)

    else:
        this_expr.print_log(dis_msg , print_=False)
        pass
        if history is not None:
            history.loc[epoch, 'eval_total_loss'] = running_loss
            # history.loc[epoch, 'eval_BEV_loss'] = running_BEV
            # history.loc[epoch, 'eval_pts_3D_loss'] = running_pts_3D

            history.loc[epoch, 'eval_time'] = time.time() - ep_since

        # with open("evaltion_Pack.pandas", 'wb') as handle:
            # pickle.dump(evaltion_Pack, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return  running_loss ,history
