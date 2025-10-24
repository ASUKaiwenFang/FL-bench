## The scale of noise

We use the following formula to calculate the scale of noise:

$$
\sigma_{DP} = \frac{2 \cdot C \cdot \sigma_g}{b}
$$

Also, we clip the gradient by the norm, which means $$g_{i} = \frac{g_{i}}{\max \{1, \ \|g_{i}\|/\mathcal{C}\}}$$ So the ratio between the noisy gradient and the clipped gradient shoule be in some range.

Please check this
```
2025-10-22 18:36:24,179 - src.client.dp_fedavg_local - INFO - [base.0.weight] mean_clipped_grad: shape=torch.Size([200, 784]), norm=3.337447, mean=-0.000055, std=0.008428, min=-0.097662, max=0.076486
2025-10-22 18:36:24,179 - src.client.dp_fedavg_local - INFO - [base.0.weight] noisy_grad: shape=torch.Size([200, 784]), norm=973.550842, mean=-0.004037, std=2.458591, min=-10.228232, max=10.709750
2025-10-22 18:36:24,179 - src.client.dp_fedavg_local - INFO - [base.0.bias] mean_clipped_grad: shape=torch.Size([200]), norm=0.163740, mean=-0.001036, std=0.011561, min=-0.042005, max=0.031569
2025-10-22 18:36:24,179 - src.client.dp_fedavg_local - INFO - [base.0.bias] noisy_grad: shape=torch.Size([200]), norm=1.108243, mean=-0.004411, std=0.078437, min=-0.213237, max=0.194440
2025-10-22 18:36:24,180 - src.client.dp_fedavg_local - INFO - [base.2.weight] mean_clipped_grad: shape=torch.Size([200, 200]), norm=4.299016, mean=-0.000762, std=0.021482, min=-0.226263, max=0.208860
2025-10-22 18:36:24,180 - src.client.dp_fedavg_local - INFO - [base.2.weight] noisy_grad: shape=torch.Size([200, 200]), norm=448.354584, mean=-0.011337, std=2.241772, min=-9.215239, max=10.000916
2025-10-22 18:36:24,180 - src.client.dp_fedavg_local - INFO - [base.2.bias] mean_clipped_grad: shape=torch.Size([200]), norm=0.363520, mean=-0.001288, std=0.025737, min=-0.069586, max=0.063749
2025-10-22 18:36:24,180 - src.client.dp_fedavg_local - INFO - [base.2.bias] noisy_grad: shape=torch.Size([200]), norm=2.140428, mean=-0.012879, std=0.151181, min=-0.442712, max=0.336408
2025-10-22 18:36:24,181 - src.client.dp_fedavg_local - INFO - [classifier.weight] mean_clipped_grad: shape=torch.Size([10, 200]), norm=6.489855, mean=0.000000, std=0.145154, min=-1.103498, max=0.231059
2025-10-22 18:36:24,181 - src.client.dp_fedavg_local - INFO - [classifier.weight] noisy_grad: shape=torch.Size([10, 200]), norm=133.819763, mean=0.042880, std=2.992742, min=-13.319304, max=9.788938
2025-10-22 18:36:24,181 - src.client.dp_fedavg_local - INFO - [classifier.bias] mean_clipped_grad: shape=torch.Size([10]), norm=0.769276, mean=-0.000000, std=0.256425, min=-0.722560, max=0.150120
2025-10-22 18:36:24,181 - src.client.dp_fedavg_local - INFO - [classifier.bias] noisy_grad: shape=torch.Size([10]), norm=1.317783, mean=-0.018557, std=0.438825, min=-0.797782, max=0.665157
```

We can see, to maintain the privacy budget, the noise is not small.
**Q: Is that the requirement of the dp? Because attackers can not get any information about the data.**

If we check the paper, the DP-SGD is bad while DP-FedAvg is good. Why using DP-Fedavg is better? Because eliminating the effect of the noise in aggregation step?


## JSE

### The shrinkage factor
$$
shrinkage factor = \left( 1 - \frac{(d-2)\sigma^2}{\|a\|^2} \right)
$$
which is a number. If we use it to scale the gradient, the role is similar to the learning rate. Why should we use it?

Suppose we want this number decrease gradully from 1 to some small number, why not use a decay learning rate?

Another perspective is that, considering the noise, the best learning rate is not stable at every step, and the variance of the best learning rate is large. So JSE is a kind of adaptive learning rate. 

But from the experiment, we find that usually the shrinkage factor is usually small and stable. So it does not meet this guess.

### per layer JSE or per feature JSE

If we consider per layer JSE or per feature JSE, actually it adjusts the direction of the gradient. So it's no longer explained as the adaptive learning rate.


### Move back: the benefit of JSE
If we consider the benefit of JSE, it has a smaller MSE. Recall that,
$$E_\theta [\|\hat \theta - \theta\|^2] = E_\theta [\| \hat \theta - E_\theta[\hat \theta]\|^2] + (E_\theta[\hat \theta] - \theta)^2$$
which means MSE = Variance + Bias^2.

The reason why the JSE shrinkage factor is small is that the variance of $\hat \theta$ is large. So JSE shrink it to 0 to reduce the variance while introduce a larger bias. Why should we focus on improving the total MSE here? I suppose some gradient is much more important than others. If we do in this way, this gradient will be shrinked to 0.

## algorithm

For fedavg, we have $\bar g = \frac{1}{S} \sum_{i=1}^S g_i$. 

How about using JSE here? Likely, we have $$g_i^{JSE} = \bar g + \left( 1 - \frac{(d-2)\sigma^2}{\|g_i - \bar g\|^2} \right) (g_i - \bar g)$$ and $$\bar g^{JSE} = \frac{1}{S} \sum_{i=1}^S g_i^{JSE}$$.

This is my feeling and I'm still stuck here. 

Here, if the $g_i$ is far away from the $\bar g$, the shrinkage factor is close to 1, which means no shrinkage. If the $g_i$ is close to the $\bar g$, the shrinkage factor is close to 0, which means a lot of shrinkage. Could that say we maintain some heterogeneity of the gradients?

