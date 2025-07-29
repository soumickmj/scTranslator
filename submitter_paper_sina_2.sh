#!/bin/bash

# --- Job Submission Manager ---
# This script submits a list of sbatch jobs, ensuring that no more than
# a specified maximum number of jobs from this specific script are running
# or pending at the same time.

# Set the maximum number of concurrent jobs
MAX_JOBS=9

# Array to hold all the sbatch commands
# Note: Each command is a single string element in the array.
COMMANDS=(
  'sbatch -J scV2_2M_AliceFT_1ep /ssu/gassu/software/scTranslator/newV/scTranslator/launcher.sh --args " --pretrain_checkpoint checkpoint/stage2p5_scV2_2M_Alice_20250528_AliceFT_1ep.pt --tag_test scV2_2M_Alice_20250528_AliceFT_1ep_LasryTst --RNA_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_RNA_mapped_scTranslator.h5ad --Pro_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_ADT_zero_mapped_scTranslator.h5ad"'
  'sbatch -J scV2_2M_AliceFT_5ep /ssu/gassu/software/scTranslator/newV/scTranslator/launcher.sh --args " --pretrain_checkpoint checkpoint/stage2p5_scV2_2M_Alice_20250528_AliceFT_5ep.pt --tag_test scV2_2M_Alice_20250528_AliceFT_5ep_LasryTst --RNA_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_RNA_mapped_scTranslator.h5ad --Pro_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_ADT_zero_mapped_scTranslator.h5ad"'
  'sbatch -J scV2_2M_woFT_LasryTst /ssu/gassu/software/scTranslator/newV/scTranslator/launcher.sh --args " --tag_test scV2_2M_Alice_20250528_woFT_LasryTst --RNA_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_RNA_mapped_scTranslator.h5ad --Pro_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_ADT_zero_mapped_scTranslator.h5ad --pretrain_checkpoint checkpoint/stage2_scTranslatorV2_2M.pt"'
  'sbatch -J scV2_160k_AliceFT_1ep /ssu/gassu/software/scTranslator/newV/scTranslator/launcher.sh --args " --pretrain_checkpoint checkpoint/stage2p5_scV2_160k_Alice_20250528_AliceFT_1ep.pt --tag_test scV2_160k_Alice_20250528_AliceFT_1ep_LasryTst --RNA_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_RNA_mapped_scTranslator.h5ad --Pro_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_ADT_zero_mapped_scTranslator.h5ad"'
  'sbatch -J scV2_160k_AliceFT_5ep /ssu/gassu/software/scTranslator/newV/scTranslator/launcher.sh --args " --pretrain_checkpoint checkpoint/stage2p5_scV2_160k_Alice_20250528_AliceFT_5ep.pt --tag_test scV2_160k_Alice_20250528_AliceFT_5ep_LasryTst --RNA_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_RNA_mapped_scTranslator.h5ad --Pro_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_ADT_zero_mapped_scTranslator.h5ad"'
  'sbatch -J scV2_160k_woFT_LasryTst /ssu/gassu/software/scTranslator/newV/scTranslator/launcher.sh --args " --tag_test scV2_160k_Alice_20250528_woFT_LasryTst --RNA_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_RNA_mapped_scTranslator.h5ad --Pro_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_ADT_zero_mapped_scTranslator.h5ad --pretrain_checkpoint checkpoint/stage2_scTranslatorV2_160K.pt"'
  'sbatch -J scV2_10k_AliceFT_1ep /ssu/gassu/software/scTranslator/newV/scTranslator/launcher.sh --args " --pretrain_checkpoint checkpoint/stage2p5_scV2_10k_Alice_20250528_AliceFT_1ep.pt --tag_test scV2_10k_Alice_20250528_AliceFT_1ep_LasryTst --RNA_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_RNA_mapped_scTranslator.h5ad --Pro_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_ADT_zero_mapped_scTranslator.h5ad"'
  'sbatch -J scV2_10k_AliceFT_5ep /ssu/gassu/software/scTranslator/newV/scTranslator/launcher.sh --args " --pretrain_checkpoint checkpoint/stage2p5_scV2_10k_Alice_20250528_AliceFT_5ep.pt --tag_test scV2_10k_Alice_20250528_AliceFT_5ep_LasryTst --RNA_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_RNA_mapped_scTranslator.h5ad --Pro_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_ADT_zero_mapped_scTranslator.h5ad"'
  'sbatch -J scV2_10k_woFT_LasryTst /ssu/gassu/software/scTranslator/newV/scTranslator/launcher.sh --args " --tag_test scV2_10k_Alice_20250528_woFT_LasryTst --RNA_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_RNA_mapped_scTranslator.h5ad --Pro_path /scratch/sina.kanannejad/scTrnas_Sina_prep/Lasry_ADT_zero_mapped_scTranslator.h5ad --pretrain_checkpoint checkpoint/stage2_scTranslatorV2_10K.pt"'
)

# Array to hold the job IDs submitted by this script
SUBMITTED_JOB_IDS=()

# Counter for submitted jobs
submitted_count=0
total_jobs=${#COMMANDS[@]}

# Loop through and submit commands
for cmd in "${COMMANDS[@]}"; do
    # This inner loop will continue until a job is successfully submitted
    while true; do
        # Get the number of jobs from our list that are still running or pending
        job_list=$(IFS=,; echo "${SUBMITTED_JOB_IDS[*]}")
        
        #To count all the jobs from the user, not only from this script
        # CURRENT_JOBS=$(squeue -u "$USER" -h -t RUNNING,PENDING | wc -l)
        
        if [ -z "$job_list" ]; then
            CURRENT_JOBS=0
        else
            # The '-h' flag removes the header from squeue output
            CURRENT_JOBS=$(squeue -h -j "$job_list" -t RUNNING,PENDING | wc -l)
        fi
        
        if [ "$CURRENT_JOBS" -lt "$MAX_JOBS" ]; then
            echo "Current script jobs ($CURRENT_JOBS) are below the limit of $MAX_JOBS."
            echo "Submitting job $(($submitted_count + 1)) of ${total_jobs}..."
            
            # Use 'eval' to execute the command string and capture the output
            output=$(eval "$cmd")
            # Extract the job ID from the sbatch output ("Submitted batch job 12345")
            job_id=$(echo "$output" | awk '{print $4}')
            
            # Add the new job ID to our list for tracking
            SUBMITTED_JOB_IDS+=("$job_id")
            echo "Submission successful. Job ID: $job_id"
            
            # Increment submitted job counter
            ((submitted_count++))
            echo "Total submitted: $submitted_count / $total_jobs."
            
            # Break the inner 'while' loop to move to the next command
            break
        else
            echo "Max jobs ($MAX_JOBS) from this script reached. Currently at $CURRENT_JOBS. Waiting..."
            # Wait for a minute before checking again
            sleep 1000
        fi
    done
done

echo "All ${total_jobs} jobs have been submitted to the queue."
