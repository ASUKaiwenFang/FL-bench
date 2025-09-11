# Introduction to privacy settings in DP-SCAFFOLD
    The reason why we cannot use the Opacus is that they can onlyadd noise to the gradients, warpped in the optimizer.
# Introduction to this repo
 - We only need to consider Fedavg server class and client class.
 - The left part we can ignore.

# something need to be decided

- For Fedavg, we use all batches in one round. For DP-Fedavg, they use one mini-batch in one round. Should we use all batches in one round or one mini-batch in one round?
- For choosing one mini-batch in one round, no matter which algorithm we use, do we need to totally random it or we can just generate the order first and then choose the mini-batch iteratively(Round-robin)?
```python
    np.random.seed(500 * (self.times + 1) * (glob_iter + 1) + epoch + 1)
    torch.manual_seed(500 * (self.times + 1) * (glob_iter + 1) + epoch + 1)
    train_idx = np.arange(self.train_samples)
    train_sampler = SubsetRandomSampler(train_idx)
    self.trainloader = DataLoader(self.train_data, self.batch_size, sampler=train_sampler)
```
vs
```python
    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
            if len(x) <= 1:
                x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)
        return x.to(self.device), y.to(self.device)
```
- For our code, do we need to consider to communicate the parameters to the server? Or just the pseudo gradients? 
- How many algorithm do we need to implement? 
    - Fedavg, DP-Fedavg, Scaffold, DP-Scaffold, DP-FedStein, DP-ScaffoldStein
    - For DP, where to add what kind of noise?
    - For JSE, where to implement it?