import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy
from tqdm import tqdm
import pandas as pd
import sys
import random
import numpy as np
from torch.utils.data import Dataset

#################################################
#------------ Train & Test Function ------------#
#################################################   
def setup_seed(seed):
    #--- Fix random seed ---#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss2 = nn.CosineSimilarity(dim=0, eps=1e-8)
    train_loss = 0
    train_ccc = 0
    for idx, (x, y) in enumerate(tqdm(train_loader)):
        #--- Extract Feature ---#
        RNA_geneID = torch.tensor(x[:,1].tolist()).long().to(device)
        Protein_geneID = torch.tensor(y[:,1].tolist()).long().to(device)
        rna_mask = torch.tensor(x[:,2].tolist()).bool().to(device)
        pro_mask = torch.tensor(y[:,2].tolist()).bool().to(device)
        x = torch.tensor(x[:,0].tolist(), dtype=torch.float32).to(device)
        y = torch.tensor(y[:,0].tolist(), dtype=torch.float32).to(device)

        #--- Prediction ---#
        optimizer.zero_grad()
        _, y_hat = model(x, RNA_geneID, Protein_geneID, enc_mask=rna_mask, dec_mask=pro_mask)

        #--- Compute Performance Metric ---#
        y_hat = torch.squeeze(y_hat)
        y_hat = torch.where(torch.isnan(y), torch.full_like(y_hat, 0), y_hat)
        y = torch.where(torch.isnan(y), torch.full_like(y, 0), y)

        loss = F.mse_loss(y_hat[pro_mask], y[pro_mask])
        loss_step = loss.item()
        train_loss += loss_step
        
        ccc_step = loss2(y_hat[pro_mask], y[pro_mask]).item()
        train_ccc += ccc_step
        loss.backward()
        optimizer.step()

        args.wandbrun.log({"loss_step": loss_step, "ccc_step": ccc_step}, step=(epoch*len(train_loader))+idx)
    
    train_loss /= len(train_loader)
    train_ccc /= len(train_loader)
    print('-'*15)
    print('--- Epoch {} ---'.format(epoch), flush=True)
    print('-'*15)
    print('Training set: Average loss: {:.4f}, Average ccc: {:.4f}'.format(train_loss, train_ccc), flush=True)
    return train_loss, train_ccc
    
def test(model, device, test_loader):
    model.eval()
    loss2 = nn.CosineSimilarity(dim=0, eps=1e-8)
    test_loss = 0
    test_ccc = 0
    y_hat_all = []
    y_all = []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            if x.shape[-1] > 20000 or y.shape[-1] > 1000: #TODO: make  them flags
                if test_loader.batch_size != 1:
                    sys.exit("test currently implemented only for batch size 1, if x>20000 or y>1000")
                x, y = create_sliding_window_batches(x, y, 20000, 1000) 

            #--- Extract Feature ---#
            RNA_geneID = torch.tensor(x[:,1].tolist()).long().to(device)
            Protein_geneID = torch.tensor(y[:,1].tolist()).long().to(device)
            rna_mask = torch.tensor(x[:,2].tolist()).bool().to(device)
            pro_mask = torch.tensor(y[:,2].tolist()).bool().to(device)
            x = torch.tensor(x[:,0].tolist(), dtype=torch.float32).to(device)
            y = torch.tensor(y[:,0].tolist(), dtype=torch.float32).to(device)

            #--- Prediction ---#
            _, y_hat = model(x, RNA_geneID, Protein_geneID, enc_mask=rna_mask, dec_mask=pro_mask)

            #--- Compute Performance Metric ---#
            y_hat = torch.squeeze(y_hat)
            y_hat = torch.where(torch.isnan(y), torch.full_like(y_hat, 0), y_hat)
            y = torch.where(torch.isnan(y), torch.full_like(y, 0), y)
            test_loss += F.mse_loss(y_hat[pro_mask], y[pro_mask]).item()
            test_ccc += loss2(y_hat[pro_mask], y[pro_mask]).item()

            if device == 'cpu':
                y_hat_all.extend(y_hat[pro_mask].view(y_hat.shape[0], -1).numpy().tolist())
                y_all.extend(y[pro_mask].view(y_hat.shape[0], -1).numpy().tolist())
            else:
                y_hat_all.extend(y_hat[pro_mask].view(y_hat.shape[0], -1).detach().cpu().numpy().tolist())
                y_all.extend(y[pro_mask].view(y_hat.shape[0], -1).detach().cpu().numpy().tolist())
       

    test_loss /= len(test_loader)
    test_ccc /= len(test_loader)
    return test_loss, test_ccc, np.array(y_hat_all), np.array(y_all)
    
#################################################
#---------- Dataset Preprocess Function ---------#
#################################################
def normalization(x, low=1e-8, high=1):
    MIN = min(x)
    MAX = max(x)
    x = low + (x-MIN)/(MAX-MIN)*(high-low) # zoom to (low, high)
    return x

def fix_sc_normalize_truncate_padding(x, length):
    '''
    x = (num_gene,1)

    '''
    len_x = len(x.X[0])
    tmp = [i for i in x.X[0]]
    tmp = normalization(tmp)
    if len_x >= length: # truncate
        x_value = tmp[:length]
        gene = x.var.iloc[:length]['my_Id'].astype(int).values.tolist()
        mask = np.full(length, True).tolist()
    else: # padding
        x_value = tmp.tolist()
        x_value.extend([0 for i in range(length-len_x)])
        gene = x.var['my_Id'].astype(int).values.tolist()
        gene.extend([0 for i in range(length-len_x)])
        mask = np.concatenate((np.full(len_x,True), np.full(length-len_x,False)))
    return x_value, gene, mask

class fix_SCDataset(Dataset):
    def __init__(self, scRNA_adata, scP_adata, len_rna, len_protein):
        super().__init__()
        self.scRNA_adata = scRNA_adata
        self.scP_adata = scP_adata
        self.len_rna = len_rna
        self.len_protein = len_protein

    def __getitem__(self, index):
        k = self.scRNA_adata.obs.index[index]
        rna_value, rna_gene, rna_mask = fix_sc_normalize_truncate_padding(self.scRNA_adata[k], self.len_rna)
        pro_value, pro_gene, pro_mask = fix_sc_normalize_truncate_padding(self.scP_adata[k], self.len_protein)
        return np.array([rna_value, rna_gene, rna_mask]), np.array([pro_value, pro_gene, pro_mask])

    def __len__(self):
        return self.scRNA_adata.n_obs

def sc_normalize_truncate_padding(x, length):
    '''
    x = (num_gene,1)

    '''
    len_x = len(x.X[0])
    tmp = [i for i in x.X[0]]
    tmp = normalization(tmp)
    if len_x >= length: # truncate
        gene = random.sample(range(len_x), length)
        x_value = [i for i in tmp[gene]] 
        gene = x.var.iloc[gene]['my_Id'].astype(int).values.tolist()
        mask = np.full(length, True).tolist()
    else: # padding
        x_value = tmp.tolist()
        x_value.extend([0 for i in range(length-len_x)])
        gene = x.var['my_Id'].astype(int).values.tolist()
        gene.extend([0 for i in range(length-len_x)])
        mask = np.concatenate((np.full(len_x,True), np.full(length-len_x,False)))
    return x_value, gene, mask

class SCDataset(Dataset):
    def __init__(self, scRNA_adata, scP_adata, len_rna, len_protein):
        super().__init__()
        self.scRNA_adata = scRNA_adata
        self.scP_adata = scP_adata
        self.len_rna = len_rna
        self.len_protein = len_protein

    def __getitem__(self, index):
        k = self.scRNA_adata.obs.index[index]
        rna_value, rna_gene, rna_mask = sc_normalize_truncate_padding(self.scRNA_adata[k], self.len_rna)
        pro_value, pro_gene, pro_mask = sc_normalize_truncate_padding(self.scP_adata[k], self.len_protein)
        return np.array([rna_value, rna_gene, rna_mask]), np.array([pro_value, pro_gene, pro_mask])

    def __len__(self):
        return self.scRNA_adata.n_obs
    
def attention_normalize(weights):
    for i in weights.columns:
        W_min = weights[i].min()
        W_max = weights[i].max()
        weights[i] = (weights[i]-W_min)/(W_max-W_min)
    for i in range(weights.shape[0]):
        W_min = weights.iloc[i].min()
        W_max = weights.iloc[i].max()
        weights.iloc[i] = (weights.iloc[i]-W_min)/(W_max-W_min)
    return(weights)

#################################################
#---- Additional functions added by Soumick ----#
#################################################

#update the input data files, to handle any custom file instead of a the strict format of this project
def update_adata(adata, args):
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    if args.id_col:
        adata.var = adata.var.rename({args.id_col: "my_Id"}, axis=1)
    if args.index_col:
        adata.var = adata.var.reset_index().set_index(args.index_col)
        if args.old_index_col_name:
            adata.var = adata.var.rename({"index": args.old_index_col_name}, axis=1)
    if args.filter_noIDs:
        adata = adata[:, adata.var['my_Id'] != -1]
    return adata

#Slinding window to handle large input data
def create_sliding_window_batches(input_tensor, target_tensor, max_input_window, target_batch_size):
    num_target_batches = -(-target_tensor.shape[-1] // target_batch_size)  # Ceiling division
    num_input_batches = -(-input_tensor.shape[-1] // max_input_window)  # Ceiling division
    if num_target_batches == 1:
        step_size = input_tensor.shape[-1] - max_input_window
    else:
        step_size = -(-(input_tensor.shape[-1] - max_input_window) // (num_target_batches-1)) 

    input_batches = []
    target_batches = []

    for i in range(max(num_target_batches, num_input_batches)):
        if num_target_batches == 1:
            target_start_idx = 0
            target_end_idx = target_tensor.shape[-1]
        else:
            target_start_idx = i * target_batch_size
            target_end_idx = min((i + 1) * target_batch_size, target_tensor.shape[-1])
        
        target_batch = target_tensor[...,target_start_idx:target_end_idx]
        if target_batch.shape[-1] < target_batch_size:
            pad_size = list(target_batch.shape)
            pad_size[-1] = target_batch_size - target_batch.shape[-1]
            target_batch = torch.cat([target_batch, torch.zeros(pad_size, dtype=target_batch.dtype, device=target_batch.device)], dim=-1)
        target_batches.append(target_batch)
        
        input_start_idx = max(0, min(i * step_size, target_start_idx))
        if input_start_idx+max_input_window > input_tensor.shape[-1]:
            input_start_idx = max(0, input_tensor.shape[-1] - max_input_window)
        input_end_idx = min(input_start_idx + max_input_window, input_tensor.shape[-1])
        if input_end_idx < target_end_idx:
            input_end_idx = min(input_tensor.shape[-1], target_end_idx)

        input_batch = input_tensor[...,input_start_idx:input_end_idx]
        input_batches.append(input_batch)

    return torch.cat(input_batches, dim=0), torch.cat(target_batches, dim=0)