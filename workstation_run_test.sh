source .venv/bin/activate

EXPERIMENT_VERSION="dp_scaffold_1"
EXPERIMENT_DESCRIPTION="DP-FedAvg-Local experiments with sigma values: 0.1, 0.5, 1, 10 and lr values: 0.01, 0.05, 0.1"

# Define directory variables
OUTPUT_DIR="experiments/${EXPERIMENT_VERSION}"
LOG_DIR="${OUTPUT_DIR}/logs"


# Check if experiment directory already exists
if [ -d "${OUTPUT_DIR}" ]; then
    echo "Error: Experiment directory ${OUTPUT_DIR} already exists. Please use a different EXPERIMENT_VERSION or remove the existing directory."
    exit 1
fi

# Create necessary directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Create experiment metadata markdown file
cat > "${OUTPUT_DIR}/experiment_metadata.md" << EOF
# Experiment Metadata

- **Experiment Version**: ${EXPERIMENT_VERSION}
- **Experiment Description**: ${EXPERIMENT_DESCRIPTION}
- **Date**: $(date "+%Y-%m-%d %H:%M:%S")
- **Working Directory**: $(pwd)
EOF

max_concurrent=6  
current_jobs=0
total_gpus=2

# For testing
# config_files=("dp_fedavg_step_noise" "dp_fedavg_last_noise")
config_files=("dp_scaffold_step_noise" "dp_scaffold_last_noise")
# config_files=("dp_fedstein_step_noise_step_jse" "dp_fedstein_step_noise_final_jse" "dp_fedstein_last_noise_server_jse")
# config_files=("dp_scaffstein_step_noise_step_jse" "dp_scaffstein_step_noise_final_jse" "dp_scaffstein_last_noise_server_jse")

sigma_values=(0.1 0.5 1 10)
lr_values=(0.01 0.05 0.1)
global_epoch_values=(200)
local_epoch_values=(5 10 15)

echo "Starting parallel experiments with max $max_concurrent concurrent jobs..."
echo "Using $total_gpus GPUs"

for config_file in "${config_files[@]}"; do
    for sigma in "${sigma_values[@]}"; do
        for lr in "${lr_values[@]}"; do
            for global_epoch in "${global_epoch_values[@]}"; do
                for local_epoch in "${local_epoch_values[@]}"; do

                    while (( $(jobs -r | wc -l) >= max_concurrent )); do
                        echo "Waiting for running jobs to finish... ($(jobs -r | wc -l)/$max_concurrent)"
                        sleep 10
                    done

                    gpu_id=$((current_jobs % total_gpus))

                    
                    if [[ $config_file == dp_fedavg* ]]; then
                        method_param="dp_fedavg_local"
                    elif [[ $config_file == dp_scaffold_* ]]; then
                        method_param="dp_scaffold"
                    elif [[ $config_file == dp_fedstein* ]]; then
                        method_param="dp_fed_stein"
                    elif [[ $config_file == dp_scaffstein* ]]; then
                        method_param="dp_scaffstein"
                    fi

                    exp_name="${config_file}_sigma_${sigma}_lr_${lr}_global_epoch_${global_epoch}_local_epoch_${local_epoch}"

                    echo "[$(date '+%H:%M:%S')] Starting experiment $((current_jobs+1)): ${config_file} on GPU ${gpu_id}"

                    CUDA_VISIBLE_DEVICES=${gpu_id} python experiment.py \
                        --config-name ${config_file} \
                        ${method_param}.sigma=${sigma} \
                        optimizer.lr=${lr} \
                        common.global_epoch=${global_epoch} \
                        common.local_epoch=${local_epoch} \
                        hydra.run.dir=${OUTPUT_DIR}/${exp_name} \
                        > ${LOG_DIR}/${exp_name}.log 2>&1 &

                    ((current_jobs++))
                    sleep 2  
                done
            done
        done
    done
done

echo "All experiments submitted! Waiting for completion..."
wait
echo "[$(date '+%H:%M:%S')] All experiments completed!"
echo "Check logs in: ${LOG_DIR}/"
echo "Check results in: ${OUTPUT_DIR}/"