#!/bin/bash
#SBATCH --account=m5073_g
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --job-name=dp_fedstein_step_noise_step_jse_sigma_0.3_lr_10.0_global_epoch_100_local_epoch_50_clip_norm_10.0_clip_method_heuristic_data_sample_ratio_0.1
#SBATCH --output=experiments/dp_fedstein_step_noise_step_jse_positive_perlayer_jse/logs/dp_fedstein_step_noise_step_jse_sigma_0.3_lr_10.0_global_epoch_100_local_epoch_50_clip_norm_10.0_clip_method_heuristic_data_sample_ratio_0.1.log
#SBATCH --error=experiments/dp_fedstein_step_noise_step_jse_positive_perlayer_jse/logs/dp_fedstein_step_noise_step_jse_sigma_0.3_lr_10.0_global_epoch_100_local_epoch_50_clip_norm_10.0_clip_method_heuristic_data_sample_ratio_0.1.log
module load conda
cd FL-bench
conda activate fl-bench-home
python experiment.py --config-name dp_fedstein_step_noise_step_jse dp_fed_stein.sigma=0.3 dp_fed_stein.clip_norm=10.0 dp_fed_stein.clip_method=heuristic dp_fed_stein.data_sample_ratio=0.1 optimizer.lr=10.0 common.global_epoch=100 common.local_epoch=50 hydra.run.dir=experiments/dp_fedstein_step_noise_step_jse_positive_perlayer_jse/dp_fedstein_step_noise_step_jse_sigma_0.3_lr_10.0_global_epoch_100_local_epoch_50_clip_norm_10.0_clip_method_heuristic_data_sample_ratio_0.1
