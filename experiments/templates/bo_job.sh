#!/bin/bash

#SBATCH --job-name=bo-{job_name}
#SBATCH --array=0-{n_trials}
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --mem={mem}
#SBATCH --output={log_dir}/%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=waltervirany@gmail.com

module purge
module load python
module load anaconda
conda activate molcollisions

python bo.py --target {target} --fp_config {fp_config} --pool {pool} --n_init {n_init} --budget {budget} --acq_func {acq_func}{save_results_flag}
