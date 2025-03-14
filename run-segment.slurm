#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu-p40_default
#SBATCH --gres=gpu:p40:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name="[Corallian] Segmentation Pipeline"
#SBATCH --output="run-segmentation.out" 
#SBATCH --error="run-segmentation.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jana_bantolino@dlsu.edu.ph

## For more `sbatch` options, use `man sbatch` in the HPC, or go to https://slurm.schedmd.com/sbatch.html.

## Set stack size to unlimited.
ulimit -s unlimited

## Benchmarking.
start_time=$(date +%s.%N)

## Print job parameters.
echo "Submitted on $(date)"
echo "JOB PARAMETERS"
echo "SLURM_JOB_ID          : ${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME        : ${SLURM_JOB_NAME}"
echo "SLURM_JOB_NUM_NODES   : ${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_NODELIST    : ${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS          : ${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE : ${SLURM_NTASKS_PER_NODE}"
echo "SLURM_MEM_PER_NODE    : ${SLURM_MEM_PER_NODE}"

## Create a unique temporary folder in the node. Using a local temporary folder usually results in faster read/write for temporary files.
custom_tmpdir="yes"

if [[ $custom_tmpdir == "yes" ]]; then
   JOB_TMPDIR=/tmp/${USER}/SLURM_JOB_ID/${SLURM_JOB_ID}
   mkdir -p ${JOB_TMPDIR}
   export TMPDIR=${JOB_TMPDIR}
   echo "TMPDIR                : $TMPDIR"
fi

## Reset modules.
module purge
module load anaconda

## Main job. Run your codes and executables here; `srun` is optional.
## Checking
echo "Activating corallian-env..."
conda activate corallian-env
echo "==============================="
echo ""

## Check Python version and location
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "==============================="
echo ""

## List all installed packages using conda
# echo "Installed packages (conda list):"
# conda list

## List all installed packages using pip
# echo "Installed packages (pip list):"
# python -m pip list

echo "Running debug-train.py..."
python debug-train.py
echo "==============================="
echo ""

echo "Running segment_images.py..."
python -u segment_images.py
conda deactivate

## Flush the TMPDIR.
if [[ $custom_tmp == "yes" ]]; then
   rm -rf $TMPDIR
   echo "Cleared the TMPDIR (${TMPDIR})"
fi

## Benchmarking
end_time=$(date +%s.%N)
echo "Finished on $(date)"
run_time=$(python -c "print($end_time - $start_time)")
echo "Total runtime (sec): ${run_time}"