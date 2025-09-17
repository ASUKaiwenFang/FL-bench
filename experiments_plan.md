dataset: **MNIST**, **EMNIST**, (FEMNIST, CIFAR10)
iid: 0.0, 0.1, 1.0?
Dirichlet: 0.0, 0.1, 1.0?
model: 2NN, CNN
method: **dp_scaffstein**, dp_fed_stein, dp_scaffold, dp_fed_avg
algorithm_variant: step_noise_step_jse, step_noise_final_jse, last_noise_server_jse
(client_number, sample_client_number): (10, 10), **(100, 10)**, (100, 20)
local_epoch: 5, 10
global_epoch: 50, 100
batch_size: 32
sigma: 0.1, 0.5, 1.0?









