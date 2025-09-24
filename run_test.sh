#!/bin/bash
# FL-bench DP-FedAvg-Local Experiments with PBS
# This script runs differential privacy federated learning experiments
# with different sigma values using PBS job scheduler

# Define variables to make the script more readable and maintainable
EXPERIMENT_VERSION="dp_fedavg_local_sigma_sweep"
EXPERIMENT_DESCRIPTION="DP-FedAvg-Local experiments with sigma values: 0.1, 0.5, 1, 10"

# Define directory variables
OUTPUT_DIR="experiments/${EXPERIMENT_VERSION}"
LOG_DIR="${OUTPUT_DIR}/logs"
JOBSCRIPT_DIR="./job_scripts"

# Check if experiment directory already exists
if [ -d "${OUTPUT_DIR}" ]; then
    echo "Error: Experiment directory ${OUTPUT_DIR} already exists. Please use a different EXPERIMENT_VERSION or remove the existing directory."
    exit 1
fi

# Create necessary directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${JOBSCRIPT_DIR}"

# Create experiment metadata markdown file
cat > "${OUTPUT_DIR}/experiment_metadata.md" << EOF
# Experiment Metadata

- **Experiment Version**: ${EXPERIMENT_VERSION}
- **Experiment Description**: ${EXPERIMENT_DESCRIPTION}
- **Method**: dp_fedavg_local
- **Sigma Values**: [0.1, 0.5, 1, 10]
- **Date**: $(date "+%Y-%m-%d %H:%M:%S")
- **Working Directory**: $(pwd)
EOF

# For testing
config_files=("dp_fedavg_step_noise" "dp_fedavg_last_noise" "dp_scaffold_step_noise" "dp_scaffold_last_noise" "dp_fedstein_step_noise_step_jse" "dp_fedstein_step_noise_final_jse" "dp_fedstein_last_noise_server_jse" "dp_scaffstein_step_noise_step_jse" "dp_scaffstein_step_noise_final_jse" "dp_scaffstein_last_noise_server_jse")
sigma_values=(0.1)
lr_values=(0.01)
global_epoch_values=(100)
local_epoch_values=(5)


# For full experiment
# config_files=("dp_fedavg_step_noise" "dp_fedavg_last_noise" "dp_scaffold_step_noise" "dp_scaffold_last_noise" "dp_fedstein_step_noise_step_jse" "dp_fedstein_step_noise_final_jse" "dp_fedstein_last_noise_server_jse" "dp_scaffstein_step_noise_step_jse" "dp_scaffstein_step_noise_final_jse" "dp_scaffstein_last_noise_server_jse")
# sigma_values=(0.1 0.5 1 10)
# lr_values=(0.01 0.05 0.1)
# global_epoch_values=(100 200 300)
# local_epoch_values=(5 10 15)

for config_file in "${config_files[@]}"; do
    for sigma in "${sigma_values[@]}"; do   
        for lr in "${lr_values[@]}"; do
            for global_epoch in "${global_epoch_values[@]}"; do
                for local_epoch in "${local_epoch_values[@]}"; do
                    JOBSCRIPT_FILE="${JOBSCRIPT_DIR}/${config_file}_sigma_${sigma}_lr_${lr}_global_epoch_${global_epoch}_local_epoch_${local_epoch}.sh"

                    # Create job script file
                    echo "#!/bin/bash" > "${JOBSCRIPT_FILE}"

                    # Add PBS directives
                    echo "#PBS -A PPFL_FM" >> "${JOBSCRIPT_FILE}"
                    echo "#PBS -k doe" >> "${JOBSCRIPT_FILE}"
                    echo "#PBS -l filesystems=home:eagle" >> "${JOBSCRIPT_FILE}"
                    echo "#PBS -l select=1:ngpus=1:gputype=A100" >> "${JOBSCRIPT_FILE}"
                    echo "#PBS -q preemptable" >> "${JOBSCRIPT_FILE}"
                    echo "#PBS -l walltime=06:00:00" >> "${JOBSCRIPT_FILE}"
                    echo "#PBS -r y" >> "${JOBSCRIPT_FILE}"
                    echo "#PBS -j oe" >> "${JOBSCRIPT_FILE}"
                    echo "#PBS -N ${config_file}_sigma_${sigma}_lr_${lr}_global_epoch_${global_epoch}_local_epoch_${local_epoch}" >> "${JOBSCRIPT_FILE}"
                    echo "#PBS -o ${LOG_DIR}/${config_file}_sigma_${sigma}_lr_${lr}_global_epoch_${global_epoch}_local_epoch_${local_epoch}.log" >> "${JOBSCRIPT_FILE}"

                    # Add environment setup
                    echo "module use /soft/modulefiles" >> "${JOBSCRIPT_FILE}"
                    echo "module load conda" >> "${JOBSCRIPT_FILE}"
                    echo "conda activate base" >> "${JOBSCRIPT_FILE}"
                    echo "cd FL-bench" >> "${JOBSCRIPT_FILE}"

                    # Determine method parameter name for parameter overrides
                    if [[ $config_file == dp_fedavg* ]]; then
                        method_param="dp_fedavg_local"
                    elif [[ $config_file == dp_scaffold_* ]]; then
                        method_param="dp_scaffold"
                    elif [[ $config_file == dp_fedstein* ]]; then
                        method_param="dp_fed_stein"
                    elif [[ $config_file == dp_scaffstein* ]]; then
                        method_param="dp_scaffstein"
                    fi

                    # Add experiment execution command
                    echo "python experiment.py --config-name ${config_file} ${method_param}.sigma=${sigma} optimizer.lr=${lr} common.global_epoch=${global_epoch} common.local_epoch=${local_epoch} hydra.run.dir=${OUTPUT_DIR}" >> "${JOBSCRIPT_FILE}"

                    # Submit job
                    qsub "${JOBSCRIPT_FILE}"

                    # Remove temporary job script
                    rm "${JOBSCRIPT_FILE}"

                    echo "Submitted job for config=${config_file}, sigma=${sigma}, lr=${lr}, global_epoch=${global_epoch}, local_epoch=${local_epoch}"
                done
            done
        done
    done
done
