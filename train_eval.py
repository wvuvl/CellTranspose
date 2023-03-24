from torch.utils.data import DataLoader
from torch import nn,no_grad, from_numpy
import time
from tqdm import tqdm, trange
import numpy as np
import os
import tifffile
from statistics import mean
from transforms import followflows, followflows3D, resize_image, padding_2D, padding_3D
from cellpose_src import transforms

def train_network(model, train_dl, val_dl, class_loss, flow_loss, optimizer, scheduler, device, n_epochs):
    train_losses = []
    val_losses = []
    start_train = time.time()

    print('Beginning network training.\n')
    for e in range(1, n_epochs + 1):
        train_epoch_losses = []
        model.train()
        print(scheduler.get_last_lr())
        for (sample_data, sample_labels) in tqdm(train_dl, desc='Training - Epoch {}/{}'.format(e, n_epochs)):

            sample_data = sample_data.float().to(device)
            sample_labels = sample_labels.float().to(device)
            optimizer.zero_grad()
            output = model(sample_data)
            mask_loss = class_loss(output, sample_labels)
            grad_loss = flow_loss(output, sample_labels)
            train_loss = mask_loss + grad_loss
            train_epoch_losses.append(train_loss.item())
            train_loss.backward()
            optimizer.step()

        if e >= 250 or n_epochs <= 10: scheduler.step()
        
        train_epoch_loss = mean(train_epoch_losses)
        train_losses.append(train_epoch_loss)
        if val_dl is not None:
            val_epoch_loss = validate_network(model, val_dl, flow_loss, class_loss, device)
            val_losses.append(val_epoch_loss)
            print('Train loss: {:.3f}; Validation loss: {:.3f}'.format(train_epoch_loss, val_epoch_loss))
        else:
            print('Train loss: {:.3f}'.format(train_epoch_loss))

    print('Train time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_train))))
    return train_losses, val_losses


def adapt_network(model: nn.Module, source_dl, target_dl, val_dl, sas_mask_loss, contrastive_flow_loss,
                  class_loss, flow_loss, train_direct, optimizer, scheduler, device, n_epochs,
                  k, gamma_1, gamma_2, n_thresh, temperature):
    train_losses = []
    val_losses = []
    print('Beginning domain adaptation.\n')

    # Assume # of target samples << # of source samples
    start_train = time.time()
    for e in range(1, n_epochs + 1):
        model.train()
        print(scheduler.get_last_lr())
        train_epoch_losses = []
        train_target_class_losses = []
        train_target_flow_losses = []
        target_dl_iter = iter(target_dl)

        for i, (source_sample_data, source_sample_labels) in enumerate(tqdm(
                source_dl, desc='Training - Epoch {}/{}'.format(e, n_epochs))):
            optimizer.zero_grad()

            source_sample_data = source_sample_data.to(device)
            source_sample_labels = source_sample_labels.to(device)
            source_output = model(source_sample_data)

            try:
                target_sample = next(target_dl_iter)
            except StopIteration:
                target_dl_iter = iter(target_dl)
                target_sample = next(target_dl_iter)
            target_sample_data = target_sample[0].to(device)
            target_sample_labels = target_sample[1].to(device)
            target_output = model(target_sample_data)
            if not train_direct:
                if e <= n_epochs/2:
                    adaptation_class_loss = sas_mask_loss(source_output[:, 0], source_sample_labels[:, 0],
                                                          target_output[:, 0], target_sample_labels[:, 0],
                                                          margin=10, gamma_1=gamma_1, lam=0.5)
                    target_class_loss = class_loss(target_output, target_sample_labels)
                    c_loss = target_class_loss + adaptation_class_loss
                    adaptation_flow_loss = contrastive_flow_loss(source_output[:, 1:], source_sample_labels,
                                                                 target_output[:, 1:], target_sample_labels,
                                                                 k=k, gamma_2=gamma_2, n_thresh=n_thresh,
                                                                 temperature=temperature)
                    target_flow_loss = flow_loss(target_output, target_sample_labels)
                    f_loss = target_flow_loss + adaptation_flow_loss
                    train_loss = c_loss + f_loss
                else:
                    target_class_loss = class_loss(target_output, target_sample_labels)
                    target_flow_loss = flow_loss(target_output, target_sample_labels)
                    train_loss = target_class_loss + target_flow_loss
            else:
                target_class_loss = class_loss(target_output, target_sample_labels)
                target_flow_loss = flow_loss(target_output, target_sample_labels)
                train_loss = target_class_loss + target_flow_loss

            train_epoch_losses.append(train_loss.item())
            train_target_class_losses.append(target_class_loss.item())
            train_target_flow_losses.append(target_flow_loss.item())
            train_loss.backward()
            optimizer.step()

        if e <= n_epochs/2:
            scheduler.step()
        train_losses.append(mean(train_epoch_losses))
        if val_dl is not None:
            val_epoch_loss = validate_network(model, val_dl, flow_loss, class_loss, device)
            val_losses.append(val_epoch_loss)
            print('Train loss: {:.3f}; Validation loss: {:.3f}'.format(mean(train_epoch_losses), val_epoch_loss))
        else:
            print('Train loss: {:.3f}'.format(mean(train_epoch_losses)))


    print('Train time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_train))))
    return train_losses, val_losses


def validate_network(model, data_loader, flow_loss, class_loss, device):
    model.eval()
    val_epoch_losses = []
    with no_grad():
        for (val_sample_data, val_sample_labels) in tqdm(data_loader, desc='Performing validation'):
            val_sample_data = val_sample_data.to(device)
            val_sample_labels = val_sample_labels.to(device)
            output = model(val_sample_data)
            grad_loss = flow_loss(output, val_sample_labels).item()
            mask_loss = class_loss(output, val_sample_labels).item()
            val_loss = grad_loss + mask_loss
            val_epoch_losses.append(val_loss)
    return mean(val_epoch_losses)


# Evaluation - due to image size mismatches, must currently be run one image at a time
def eval_network_2D(model: nn.Module, data_loader: DataLoader, device, patch_per_batch, patch_size, min_overlap, augment=False):
    
    model.eval()
    with no_grad():
        masks = []
        data_list = []
        pred_list = []
        for (sample_data, data_file, resize_measure) in tqdm(data_loader, desc='Evaluating Test Dataset'):
            sample_data = sample_data.squeeze(0).numpy()
            sample_data = resize_image(sample_data, rsz=resize_measure)
            curr_sample, set_corner, unpadded_dims, resized_dims = padding_2D(sample_data, patch_size)
            predictions = run_overlaps(model, curr_sample, batch_size=patch_per_batch, device=device, augment=augment, 
                                        patch_size=patch_size, min_overlap=float(patch_size/min_overlap))           
            yf = predictions[:,set_corner[0]:set_corner[0]+unpadded_dims[0], set_corner[1]:set_corner[1]+unpadded_dims[1]]
            
            #resizing back to the original dim     
            yf = resize_image(yf, unpadded_dims[0], unpadded_dims[1])
            sample_mask = followflows(yf)
            
            masks.append(sample_mask)
            pred_list.append(predictions[0])
            data_list.append(os.path.basename(os.path.splitext(data_file[0])[0]))
            
    return masks, pred_list, data_list

def run_overlaps(model: nn.Module, imgi, batch_size, device, augment=False, patch_size=112, min_overlap=0.1): 
    IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=patch_size, augment=augment, tile_overlap=min_overlap)
    
    model.eval()
    ny, nx, nchan, ly, lx = IMG.shape
    IMG = np.reshape(IMG, (ny*nx, nchan, ly, lx))
    niter = int(np.ceil(IMG.shape[0] / batch_size))
    
    y = np.zeros((IMG.shape[0], 3, ly, lx))
    for k in range(niter):
        irange = np.arange(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
        
        with no_grad():
            X = from_numpy(IMG[irange]).float().to(device)
            y0 = model(X).detach().cpu().numpy()
        
        y[irange] = y0.reshape(len(irange), y0.shape[-3], y0.shape[-2], y0.shape[-1])
        
    if augment:
        y = np.reshape(y, (ny, nx, 3, patch_size, patch_size))
        y = transforms.unaugment_tiles(y)
        y = np.reshape(y, (-1, 3, patch_size, patch_size))
    
    yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
    yf = yf[:,:imgi.shape[1],:imgi.shape[2]]
    return yf  


def run_3D_network(model: nn.Module,device, curr_stack, patch_size, patch_per_batch, augment=False, min_overlap=0.1):

    
    # pad image for net so Ly and Lx are divisible by 4
    curr_stack, set_corner, unpadded_dims, resized_dims = padding_3D(curr_stack, patch_size)
    Lz, nchan = curr_stack.shape[:2]
    
    # making tiles for the first slice 
    IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(curr_stack[0], bsize=patch_size, 
                                                augment=augment, tile_overlap=min_overlap)
    ny, nx, nchan, ly, lx = IMG.shape
    curr_patch_per_batch = patch_per_batch
    curr_patch_per_batch *= max(4, (patch_size**2 // (ly*lx))**0.5)
    yf0 = np.zeros((Lz, 3, curr_stack.shape[-2], curr_stack.shape[-1]), np.float32)
    
    if ny*nx > curr_patch_per_batch:
        for i in trange(Lz):
            yfi = run_overlaps(model, curr_stack[i], batch_size=patch_per_batch, device=device, augment=augment, 
                                        patch_size=patch_size, min_overlap=min_overlap)
            yf0[i] = yfi
    else:
        # run multiple slices at the same time
        ntiles = ny*nx
        nimgs = max(2, int(np.round(patch_per_batch / ntiles)))
        niter = int(np.ceil(Lz/nimgs))
        for k in trange(niter):
            IMGa = np.zeros((ntiles*nimgs, nchan, ly, lx), np.float32)
            for i in range(min(Lz-k*nimgs, nimgs)):
                IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(curr_stack[k*nimgs+i], bsize=patch_size, 
                                                                augment=augment, tile_overlap=min_overlap)
                IMGa[i*ntiles:(i+1)*ntiles] = np.reshape(IMG, (ny*nx, nchan, ly, lx))
            
            model.eval()
            with no_grad():
                X = from_numpy(IMGa).float().to(device)
                ya = model(X).detach().cpu().numpy()
            
            for i in range(min(Lz-k*nimgs, nimgs)):
                y = ya[i*ntiles:(i+1)*ntiles]
                if augment:
                    y = np.reshape(y, (ny, nx, 3, ly, lx))
                    y = transforms.unaugment_tiles(y)
                    y = np.reshape(y, (-1, 3, ly, lx))
                yfi = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
                yfi = yfi[:,:curr_stack.shape[2],:curr_stack.shape[3]]
                yf0[k*nimgs+i] = yfi
    
    # clearing out padding
    yf0 = yf0[:,:,set_corner[0]:set_corner[0]+unpadded_dims[0], set_corner[1]:set_corner[1]+unpadded_dims[1]]
    
    return yf0

def run_3D_volume(model, device, data_vol, patch_size, patch_per_batch, rescale, augment=False, min_overlap=0.1):
    axis = ('Z', 'Y', 'X')
    planes = ['YX', 'ZX', 'ZY']
    TP = [(1, 0, 2, 3), (2, 0, 1, 3), (3, 0, 1, 2)]
    RTP = [(1, 0 , 2, 3), (1, 2, 0, 3), (1, 2, 3, 0)]
    
    data_vol = data_vol.squeeze(0).numpy()
    yf = np.zeros((3, 3, data_vol.shape[1], data_vol.shape[2], data_vol.shape[3]), np.float32)
    
    for plane_idx in range(len(planes)):
        curr_stack = data_vol.copy().transpose(TP[plane_idx])
        curr_shape = curr_stack.shape
        curr_stack = resize_image(curr_stack, rsz=rescale[plane_idx])
        
        yf0 = run_3D_network(model,device, curr_stack, patch_size, patch_per_batch, augment=augment, min_overlap=min_overlap)
        
        #resizing back to the original dim     
        yf0 = resize_image(yf0, curr_shape[2], curr_shape[3])
        yf[plane_idx] = yf0.transpose(RTP[plane_idx])
    
    return yf
    
       

# Evaluation Updated - cellpose source
def eval_network_3D(model: nn.Module, data_loader: DataLoader,
                            device, patch_per_batch, patch_size, min_overlap, results_dir, anisotropy=None, augment=False):
    
    
    for (data_vol, data_file, resize_measure) in data_loader:
        
        if anisotropy is not None:
            rescale = [
            [resize_measure, resize_measure],
            [resize_measure*anisotropy, resize_measure],
            [resize_measure*anisotropy, resize_measure]
            ]
        else:
            rescale= [resize_measure] * 3 
        
        start = time.time()
        yf = run_3D_volume(model, device, data_vol, patch_size, patch_per_batch, rescale, augment=augment, min_overlap=float(patch_size/min_overlap)) 
        # 0       1     2
        # 'ZYX', 'YZX', 'XZY'
        cellprob = yf[0][0] + yf[1][0] + yf[2][0]
        dP = np.stack((yf[1][1] + yf[2][1], yf[0][1] + yf[2][2], yf[0][2] + yf[1][2]), axis=0)
        del yf, data_vol  
        mask = np.array(followflows3D(dP, cellprob,device=device))
        print(f">>> Total masks found in 3D volume in {(time.time() - start):.3f} seconds: {len(np.unique(mask))-1}")
        tifffile.imwrite(os.path.join(results_dir, 'tiff_results', os.path.basename(data_file[0])), mask)
    
            
               
