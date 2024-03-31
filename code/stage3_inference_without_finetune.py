import os
import time
import argparse
import warnings

import scanpy as sc
import numpy as np
import pandas as pd

import sys
current_dir = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(f"{current_dir}/model") 
from performer_enc_dec import *
from utils import *

def main():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--repeat', type=int, default=1,
                        help='for repeating experiments to change seed (default: 1)')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1105,
                        help='random seed (default: 1105)')
    # parser.add_argument('--enc_max_seq_len', type=int, default=-1,
    parser.add_argument('--enc_max_seq_len', type=int, default=20000,
                        help='sequence length of encoder [If -1, all the genes will be used using sliding window]')
    # parser.add_argument('--dec_max_seq_len', type=int, default=-1,
    parser.add_argument('--dec_max_seq_len', type=int, default=1000,
                        help='sequence length of decoder [If -1, all the genes will be used using sliding window]')
    parser.add_argument('--fix_set', action='store_false',
                        help='fix (aligned) or disordering (un-aligned) dataset')
    parser.add_argument('--pretrain_checkpoint', default='checkpoint/stage2_single-cell_scTranslator.pt',
                        help='path for loading the pretrain checkpoint')
    
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Number of workers to use.')  
    
    parser.add_argument('--tag_test', default='aliceDS0_noFT_rawRNA',
                        help='tag to be used to store the test results')
    
    parser.add_argument('--RNA_path', default='/ssu/gassu/shared/scTranslator/input_data/Alice_data/clean_data_from_sina/Project_files_all/RNA_all_scrublet.scTranslatorIDs.filtered.raw.h5ad',
    #parser.add_argument('--RNA_path', default='/ssu/gassu/shared/scTranslator/input_data/Alice_data/merged_macarena_filtered.scTranslate_ids.genes.h5ad',
    # parser.add_argument('--RNA_path', default='/ssu/gassu/shared/scTranslator/input_data/10k_PBMC_3p_nextgem_Chromium_X_intron_filtered_feature_bc_matrix.scTranslate_ids.h5ad',
    # parser.add_argument('--RNA_path', default='/ssu/gassu/reference_data/single_cell/10x_test_datasets/10k_Human_PBMC_3prime_v3.1/10k_PBMC_3p_nextgem_Chromium_X_intron_filtered_feature_bc_matrix.with_scTransformer_ids.h5ad',
    # parser.add_argument('--RNA_path', default='/ssu/gassu/supporting_data/scTranslator/test/dataset1/GSM5008737_RNA_finetune_withcelltype.h5ad',
                        help='path for loading the rna')
    parser.add_argument('--Pro_path', default='/ssu/gassu/shared/scTranslator/input_data/Alice_data/clean_data_from_sina/Project_files_all/ADT_All.scTranslatorIDs.filtered.h5ad',
    # parser.add_argument('--Pro_path', default='/ssu/gassu/shared/scTranslator/input_data/10k_PBMC_3p_nextgem_Chromium_X_intron_filtered_feature_bc_matrix.scTranslate_ids.h5ad',
    # parser.add_argument('--Pro_path', default='/ssu/gassu/reference_data/single_cell/10x_test_datasets/10k_Human_PBMC_3prime_v3.1/10k_PBMC_3p_nextgem_Chromium_X_intron_filtered_feature_bc_matrix.with_scTransformer_ids.h5ad',
    # parser.add_argument('--Pro_path', default='/ssu/gassu/supporting_data/scTranslator/test/dataset1/GSM5008738_protein_finetune_withcelltype.h5ad',
    # parser.add_argument('--Pro_path', default='',
                        help='path for loading the protein')
    
    parser.add_argument('--filter_noIDs', type=int, default=1,
                        help='Whether or not to filter the elements without IDs (i.e. IDs set to -1).')  
    parser.add_argument('--id_col', default="scTranslator_id",
                        help='Which column to use as ID. If blank, my_Id column will be used.')    
    parser.add_argument('--index_col', default="",
                        help='If it is not blank, then the index will be reset and this column will be used as the index column.')  
    parser.add_argument('--old_index_col_name', default="gene_ids",
                        help='Only if index_col is supplied, this will be used to rename the index column. If blank, the index column will not be renamed after restting the index.')  
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    
    ###########################
    #--- Prepare The Model ---#
    ###########################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device',device)
    model = torch.load(args.pretrain_checkpoint, map_location=torch.device(device))
    # model = model.to(device)

    ##########################
    #--- Prepare The Data ---#
    ##########################
         
    #---  Load Single Cell Data  ---#
    scRNA_adata = sc.read_h5ad(args.RNA_path)
    scRNA_adata = update_adata(scRNA_adata, args)
    print('Total number of origin RNA genes: ', scRNA_adata.n_vars)
    print('Total number of origin cells: ', scRNA_adata.n_obs)
    print('# of NAN in X', np.isnan(scRNA_adata.X).sum())
    if args.enc_max_seq_len == -1:
        args.enc_max_seq_len = scRNA_adata.n_vars
    
    #load protein data
    scP_adata = sc.read_h5ad(args.Pro_path)
    scP_adata = update_adata(scP_adata, args)
    print('Total number of origin proteins: ', scP_adata.n_vars)
    print('# of NAN in X', np.isnan(scP_adata.X).sum())
    if args.dec_max_seq_len == -1:
        args.dec_max_seq_len = scP_adata.n_vars

    #---  Seperate Training and Testing set ---#
    test_rna = scRNA_adata
    # --- Protein ---#
    test_protein = scP_adata[test_rna.obs.index]
    # #---  Construct Dataloader ---#
    if args.fix_set == True:
        my_testset = fix_SCDataset(test_rna, test_protein, args.enc_max_seq_len, args.dec_max_seq_len)
    else:
        my_testset = SCDataset(test_rna, test_protein, args.enc_max_seq_len, args.dec_max_seq_len)

    test_loader = torch.utils.data.DataLoader(my_testset, batch_size=args.test_batch_size, drop_last=False, num_workers=args.n_workers, pin_memory=True)
    print("load data ended")

    ##################
    #---  Testing ---#
    ##################
    start_time = time.time()
    test_loss, test_ccc, y_hat, y = test(model, device, test_loader)
    y_pred =  pd.DataFrame(y_hat, columns=test_protein.var.index.tolist())
    y_truth = pd.DataFrame(y, columns=test_protein.var.index.tolist())
    ##############################
    #---  Prepare for Storage ---#
    ##############################
    
    file_path = 'result/test_'+args.tag_test
    os.makedirs(file_path, exist_ok=True)

    dict = vars(args)
    filename = open(file_path+'/args'+str(args.repeat)+'.txt','w')
    for k,v in dict.items():
        filename.write(k+':'+str(v))
        filename.write('\n')
    filename.close()
 
    #---  Save the Final Results ---#
    log_path = file_path+'/performance_log'+str(args.repeat)+'.csv'
    log_all = pd.DataFrame(columns=['test_loss', 'test_ccc'])
    log_all.loc[args.repeat] = np.array([test_loss, test_ccc])
    log_all.to_csv(log_path)
    y_pred.to_csv(file_path+'/y_pred'+str(args.repeat)+'.tsv', sep='\t')
    y_truth.to_csv(file_path+'/y_truth'+str(args.repeat)+'.tsv', sep='\t')

    print('-'*40)
    print('single cell '+str(args.enc_max_seq_len)+' RNA To '+str(args.dec_max_seq_len)+' Protein on dataset'+args.tag_test)
    print('Overall performance in repeat_%d costTime: %.4fs' % ( args.repeat, time.time() - start_time))
    print('Test Set: AVG mse %.4f, AVG ccc %.4f' % (np.mean(log_all['test_loss'][:args.repeat]), np.mean(log_all['test_ccc'][:args.repeat])))
if __name__ == '__main__':
    main()