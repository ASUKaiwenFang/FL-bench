source .venv/bin/activate

source .venv/bin/activate && python demo/main_dp_scaffold_step_noise.py optimizer.lr=0.01 model.name=2nn
source .venv/bin/activate && python demo/main_dp_scaffold_step_noise.py optimizer.lr=0.005 model.name=2nn
source .venv/bin/activate && python demo/main_dp_scaffold_step_noise.py optimizer.lr=0.001 model.name=2nn
source .venv/bin/activate && python demo/main_dp_scaffold_step_noise.py optimizer.lr=0.0001 model.name=2nn

source .venv/bin/activate && python demo/main_dp_scaffstein_step_noise_step_jse.py optimizer.lr=0.5
wait
echo "All experiments completed!"