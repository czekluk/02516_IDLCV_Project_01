# #!/bin/sh
# # General options
# # –- specify queue --
# BSUB -q gpuv100
# ## -- set the job Name --
# BSUB -J deep learning in computer vision training
# ## -- ask for number of cores (default: 1) --
# BSUB -n 4
# ## -- Select the resources: 1 gpu in exclusive process mode --
# BSUB -gpu "num=1:mode=exclusive_process"
# ## -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
# BSUB -W 03:00
# request 5GB of system-memory
# BSUB -R "rusage[mem=5GB]"
# ## -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
# #BSUB -u your_email_address
# ## -- send notification at start --
# BSUB -B
# ## -- send notification at completion--
# BSUB -N
# ## -- Specify the output and error file. %J is the job-id --
# ## -- -o and -e mean append, -oo and -eo mean overwrite --
# BSUB -o gpu_%J.out
# BSUB -e gpu_%J.err
# #-- end of LSF options --

nvidia-smi
# Load the cuda module


module unload python
module load python3
unset PYTHONPATH
unset PYTHONHOME
# pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
module load matplotlib 
pip3 install tqdm 
module load cuda/11.6 
module load numpy 

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

# Define the log file
LOG_FILE="experiment_logs.txt"

# Run the main.py script and save the output to the log file
python ./src/experiments.py > "$LOG_FILE" 2>&1

# Print a message indicating that the logs have been saved
echo "Logs have been saved to $LOG_FILE"

