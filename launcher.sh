#!/bin/bash
#SBATCH --job-name=scTransTest
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=gpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
#SBATCH --time=30-00:00:0      # walltime
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=9     # number of CPUs per task (set it to greater than the number of workers, I would go for +1)
#SBATCH --gres=gpu:1          # number of GPUs (max=4)
#SBATCH --chdir=/group/glastonbury/soumick/SLURM
#SBATCH --output=gpurun_%x_%j.log
#SBATCH --mem-per-cpu=16000Mb # RAM per CPU

exec 2>&1      # send errors into stdout stream
env | grep -e MPI -e SLURM
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID  # show slurm-command and more for DBG

module load mpi
module load cuda11.7
module load cudnn8.5-cuda11.7

###function definations

#creation of argument helper
programmename=$0
function usage {
    echo ""
    echo "Logs a SLURM job with a GPU Node."
    echo ""
    echo "usage: $programmename --root string --programme string --conda string --args \"string\" or \" \""
    echo ""
    echo "  --root    string           The program root, with the trailing slash (Default: /home/soumick.chatterjee/Codes/)"
    echo "                             [If no root is to be supplied, then put quotes with a space in between as its contents]"
    echo "  --programme string         The python file to run (Default: main.py)"
    echo "  --args    string           List of command line arguments, supplied as a single string within quotes (Default: Nothing/Blank)"
    echo "                             [Example: \"--modelID 2 --dataset UKB\". These will be supplied as arguments to the programme]"
    echo "  --conda   string           Name of the Conda env (Default: torchHTBeta2)"
    echo ""
}


#function to handle the death of the script!
function die {
    printf "Script failed: %s\n\n" "$1"
    exit 1
}

###
###process the arguments

#read the different keyworded commandline arguments
while [ $# -gt 0 ]; do
    if [[ $1 == "--help" ]]; then
        usage
        exit 0
    elif [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

#set the default values for the commandline arguments
root="${root:-/ssu/gassu/software/scTranslator/newV/scTranslator/}"
programme="${programme:-code/stage3_inference_without_finetune.py}"
args="${args:-}"
conda="${conda:-/home/soumick.chatterjee/anaconda3/envs/torchHTBeta2V2}"

###
###Start of the actual script, after reading all the arguments

#Create full path of the programme
#programmePath=$root$programme

#Change the working directory to the supplied root!
cd $root

#Program path remains as the normal supplied program
programmePath=$programme

if [[ $programmePath == *.py ]]; then
    echo "Starting the exection of the Python script: $programmePath inside the conda environment $conda";
    
    #Activate conda environment
    source /home/${USER}/.bashrc
    conda activate $conda
  
    srun python $programmePath $args
    
elif [[ $programmePath == *.sh ]]; then
    echo "Starting the exection of the Bash script: $programmePath";
  
    srun bash $programmePath $args
    
else
  echo "Starting the exection of the programme: $programmePath"; 
  
  srun $programmePath $args
    
fi


#
# END of SBATCH-script
