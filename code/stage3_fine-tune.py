import os
import time
import datetime
import argparse
import warnings


import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import wandb


import torch.optim as optim
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

import sys 
current_dir = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(f"{current_dir}/model") 
from performer_enc_dec import *
from utils import *

def main():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                        help='input batch size for each GPU training (default: 1)')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=2*1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=1, metavar='M',
                        help='Learning rate step gamma (default: 1 (not used))')
    parser.add_argument('--gamma_step', type=float, default=2000,
                        help='Learning rate step (default: 2000 (not used))')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1105,
                        help='random seed (default: 1105)')
    parser.add_argument('--repeat', type=int, default=1,
                        help='for repeating experiments to change seed (default: 1)')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training. If set to -1, will not use DDP.')
    parser.add_argument('--frac_finetune_test', type=float, default=0.1,
                        help='test set ratio [only if tst_RNA_path or tst_Pro_path is blank]')
    parser.add_argument('--dim', type=int, default=128,
                        help='latend dimension of each token')
    parser.add_argument('--enc_max_seq_len', type=int, default=20000,
                        help='sequence length of encoder')
    parser.add_argument('--dec_max_seq_len', type=int, default=1000,
                        help='sequence length of decoder')
    parser.add_argument('--translator_depth', type=int, default=2,
                        help='translator depth')
    parser.add_argument('--initial_dropout', type=float, default=0.1,
                        help='sequence length of decoder')
    parser.add_argument('--enc_depth', type=int, default=2,
                        help='sequence length of decoder')
    parser.add_argument('--enc_heads', type=int, default=8,
                        help='sequence length of decoder')
    parser.add_argument('--dec_depth', type=int, default=2,
                        help='sequence length of decoder')
    parser.add_argument('--dec_heads', type=int, default=8,
                        help='sequence length of decoder')
    parser.add_argument('--fix_set', action='store_false',
                        help='fix (aligned) or disordering (un-aligned) dataset')
    parser.add_argument('--pretrain_checkpoint', default='checkpoint/stage2_single-cell_scTranslator.pt',
                        help='path for loading the pretrain checkpoint')
    parser.add_argument('--resume', default=False, help='resume training from breakpoint')
    parser.add_argument('--path_checkpoint', default='checkpoint/stage2_single-cell_scTranslator.pt',
                        help='path for loading the resume checkpoint (need specify)')
    
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Number of workers to use.')  
    
    parser.add_argument('--tag_FT', default='aliceDS0_FT_normRNA_normPro',
                        help='tag to be used to store the fine-tuned model')
    parser.add_argument('--tag_test', default='aliceDS0_FT_normRNA_normPro_test',
                        help='tag to be used to store the test results')
    
    parser.add_argument('--RNA_path', default='/ssu/gassu/shared/scTranslator/input_data/Alice_data/clean_data_from_sina/Project_files_all/fine_tuning/RNA_all_scrublet.scTranslatorIDs.filtered.train.h5ad',
    #parser.add_argument('--RNA_path', default='dataset/test/dataset1/GSM5008737_RNA_finetune_withcelltype.h5ad',
                        help='path for loading the rna')
    parser.add_argument('--Pro_path', default='/ssu/gassu/shared/scTranslator/input_data/Alice_data/clean_data_from_sina/Project_files_all/fine_tuning/ADT_All.scTranslatorIDs.filtered.train.h5ad',
    #parser.add_argument('--Pro_path', default='dataset/test/dataset1/GSM5008738_protein_finetune_withcelltype.h5ad',
                        help='path for loading the protein')    
    
    parser.add_argument('--tst_RNA_path', default='/ssu/gassu/shared/scTranslator/input_data/Alice_data/clean_data_from_sina/Project_files_all/fine_tuning/RNA_all_scrublet.scTranslatorIDs.filtered.test.h5ad',
                        help='path for loading the test rna dataset. If blank, the rna dataset will be split into train-test')
    parser.add_argument('--tst_Pro_path', default='/ssu/gassu/shared/scTranslator/input_data/Alice_data/clean_data_from_sina/Project_files_all/fine_tuning/ADT_All.scTranslatorIDs.filtered.test.h5ad',
                        help='path for loading the test protein dataset. If blank, protein dataset will be split into train-test')    
    
    parser.add_argument('--filter_noIDs', type=int, default=1,
                        help='Whether or not to filter the elements without IDs (i.e. IDs set to -1).')  
    parser.add_argument('--id_col', default="scTranslator_id",
                        help='Which column to use as ID. If blank, my_Id column will be used.')    
    parser.add_argument('--index_col', default="",
                        help='If it is not blank, then the index will be reset and this column will be used as the index column.')  
    parser.add_argument('--old_index_col_name', default="gene_ids",
                        help='Only if index_col is supplied, this will be used to rename the index column. If blank, the index column will not be renamed after restting the index.')  
    
    
    parser.add_argument('--wnbproject', default='SingleCell',
                        help='WandB: Name of the project')
    parser.add_argument('--wnbgroup', default='scTranslator',
                        help='WandB: Name of the group')
    parser.add_argument('--wnbentity', default='glastonbury-group',
                        help='WandB: Name of the entity')
    
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    #########################
    #--- Prepare for DDP ---#
    #########################
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("use_cuda: %s" % use_cuda)
    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node: %s" % ngpus_per_node)
    is_distributed = ngpus_per_node > 1
    print('seed', args.seed)
    setup_seed(args.seed)
    print(torch.__version__)
    if args.local_rank == -1:
        is_distributed = False
        args.local_rank = 0
        print("DDP Disabled!")
    else:
        # Initializes the distributed environment to help process communication
        torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=15400))
        rank = int(os.environ['RANK'])
        print('rank', rank)
        print("DDP Initialised!")
    # Each process sets the GPU it should use based on its local rank
    print("local_rank: %s" % args.local_rank)
    device = torch.device("cuda", args.local_rank)
    print(device)
    torch.cuda.set_device(args.local_rank)

    wandbrun = wandb.init(
                            name=args.tag_FT, id=args.tag_FT, 
                            project=args.wnbproject,
                            group=args.wnbgroup,
                            entity=args.wnbentity,
                            config=args,
                        )
    args.wandbrun = wandbrun

    ###########################
    #--- Prepare The Model ---#
    ###########################
    model = scPerformerEncDec(
        dim=args.dim,
        translator_depth=args.translator_depth,
        initial_dropout=args.initial_dropout,
        enc_depth=args.enc_depth,
        enc_heads=args.enc_heads,
        enc_max_seq_len=args.enc_max_seq_len,
        dec_depth=args.dec_depth,
        dec_heads=args.dec_heads,
        dec_max_seq_len=args.dec_max_seq_len
        )
    model = torch.load(args.pretrain_checkpoint)
    # Resume training from breakpoints
    if args.resume == True:
        checkpoint = torch.load(args.path_checkpoint)
        model = checkpoint['net']
        model = model.to(device)
        if is_distributed:
            print("start init process group")
            # device_ids will include all GPU devices by default
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
            print("end init process group")
        #---  Prepare Optimizer ---#
        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True) 
        optimizer.load_state_dict(checkpoint['optimizer'])
        #---  Prepare Scheduler ---#
        scheduler = StepLR(optimizer, step_size=args.gamma_step, gamma=args.gamma)
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
        model = model.to(device)
        if is_distributed:
            print("start init process group")
            # device_ids will include all GPU devices by default
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
            print("end init process group")
        #---  Prepare Optimizer ---#
        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
        #---  Prepare Scheduler ---#
        scheduler = StepLR(optimizer, step_size=args.gamma_step, gamma=args.gamma)
    ##########################
    #--- Prepare The Data ---#
    ##########################
         
    #---  Load Single Cell Data  ---#
    scRNA_adata = sc.read_h5ad(args.RNA_path)
    scRNA_adata = update_adata(scRNA_adata, args)
    scP_adata = sc.read_h5ad(args.Pro_path)
    scP_adata = update_adata(scP_adata, args)
    print('Total number of origin RNA genes: ', scRNA_adata.n_vars)
    print('Total number of origin proteins: ', scP_adata.n_vars)
    print('Total number of origin cells: ', scRNA_adata.n_obs)
    print('# of NAN in X', np.isnan(scRNA_adata.X).sum())
    if args.enc_max_seq_len == -1:
        args.enc_max_seq_len = scRNA_adata.n_vars
    if args.dec_max_seq_len == -1:
        args.dec_max_seq_len = scP_adata.n_vars
    print('# of NAN in X', np.isnan(scP_adata.X).sum())

    #---  Seperate Training and Testing set ---#
    setup_seed(args.seed+args.repeat)

    if (not bool(args.tst_RNA_path)) or (not bool(args.tst_Pro_path)):
        print("Either tst_RNA_path or tst_Pro_path is blank. Splitting the data into train and test sets.")
        train_index, test_index = next(ShuffleSplit(n_splits=1,test_size=args.frac_finetune_test).split(scRNA_adata.obs.index))
        # --- RNA ---#
        train_rna = scRNA_adata[train_index]
        test_rna = scRNA_adata[test_index]
        # --- Protein ---#
        train_protein = scP_adata[train_index]
        test_protein = scP_adata[test_index]
    else:
        print("Both tst_RNA_path and tst_Pro_path are not blank. Loading the test set from the provided paths.")
        # --- Test RNA ---#
        test_rna = sc.read_h5ad(args.tst_RNA_path)
        test_rna = update_adata(test_rna, args)
        print('[In the test set] Total number of test RNA genes: ', test_rna.n_vars)
        print('[In the test set] Total number of test cells: ', test_rna.n_obs)
        print('[In the test set] # of NAN in X', np.isnan(test_rna.X).sum())
        # --- Test Protein ---#
        test_protein = sc.read_h5ad(args.tst_Pro_path)
        test_protein = update_adata(test_protein, args)
        print('[In the test set] Total number of test proteins: ', test_protein.n_vars)
        print('[In the test set] # of NAN in X', np.isnan(test_protein.X).sum())
        # --- Train sets ---#
        train_rna = scRNA_adata
        train_protein = scP_adata
    print("load data ended")


    #---  Construct Dataloader ---#
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': args.n_workers,
                       'shuffle': False,
                       'prefetch_factor': 2,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    if args.fix_set == True:
        my_trainset = fix_SCDataset(train_rna, train_protein, args.enc_max_seq_len, args.dec_max_seq_len)
        my_testset = fix_SCDataset(test_rna, test_protein, args.enc_max_seq_len, args.dec_max_seq_len)
    else:
        my_trainset = SCDataset(train_rna, train_protein, args.enc_max_seq_len, args.dec_max_seq_len)
        my_testset = SCDataset(test_rna, test_protein, args.enc_max_seq_len, args.dec_max_seq_len)
    
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(my_testset)

        train_loader = torch.utils.data.DataLoader(my_trainset, **train_kwargs, drop_last=True)
        test_loader = torch.utils.data.DataLoader(my_testset, **test_kwargs, drop_last=False)
    else:
        train_loader = torch.utils.data.DataLoader(my_trainset, **train_kwargs, drop_last=True)
        test_loader = torch.utils.data.DataLoader(my_testset, **test_kwargs, drop_last=False)
    
    print("data ready")

    ###############################
    #---  Training and Testing ---#
    ###############################
    start_time = time.time()
    for epoch in range(start_epoch+1, args.epochs + 1):
        print("Epoch: ", epoch, "\n")
        if is_distributed:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)        
        
        train_loss, train_ccc = train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
        args.wandbrun.log({"loss_epoch": train_loss, "ccc_epoch": train_ccc}, step=epoch)
        torch.cuda.empty_cache()

    print("Fine-tuning complete. Saving the model..")
    torch.save(model, f"checkpoint/stage2p5_{args.tag_FT}.pt")
    
    print("Testing the model..")
    test_loss, test_ccc, y_hat, y = test(model, device, test_loader)
    y_pred =  pd.DataFrame(y_hat, columns=test_protein.var.index.tolist())
    y_truth = pd.DataFrame(y, columns=test_protein.var.index.tolist())

    ##############################
    #---  Prepare for Storage ---#
    ##############################
    
    # save results in the first rank
    file_path = 'result/test_'+args.tag_test
    os.makedirs(file_path, exist_ok=True)
    # save args
    if rank == 0:
        dict = vars(args)
        filename = open(file_path+'/args'+str(args.repeat)+'.txt','w')
        for k,v in dict.items():
            filename.write(k+':'+str(v))
            filename.write('\n')
        filename.close()
 
    #---  Save the Final Results ---#
    log_path = file_path+'/'+str(rank)+'_rank_log'+str(args.repeat)+'.csv'
    log_all = pd.DataFrame(columns=['train_loss', 'train_ccc', 'test_loss', 'test_ccc'])
    log_all.loc[args.repeat] = np.array([train_loss, train_ccc, test_loss, test_ccc])
    log_all.to_csv(log_path)
    y_pred.to_csv(file_path+'/y_pred'+str(args.repeat)+'.tsv', sep='\t')
    y_truth.to_csv(file_path+'/y_truth'+str(args.repeat)+'.tsv', sep='\t')

    print('-'*40)
    print('single cell '+str(args.enc_max_seq_len)+' RNA To '+str(args.dec_max_seq_len)+' Protein on dataset'+args.tag_test)
    print('Overall performance on rank_%d in repeat_%d costTime: %.4fs' % (rank, args.repeat, time.time() - start_time))
    print('Training Set: AVG mse %.4f, AVG ccc %.4f' % (np.mean(log_all['train_loss'][:args.repeat]), np.mean(log_all['train_ccc'][:args.repeat])))
    print('Test Set: AVG mse %.4f, AVG ccc %.4f' % (np.mean(log_all['test_loss'][:args.repeat]), np.mean(log_all['test_ccc'][:args.repeat])))

if __name__ == '__main__':
    main()