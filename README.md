# User Guide
The key function is to estimate $\Delta = \ln \frac{Z(\theta_1)}{Z(\theta_0)}$ is log_ratio_normalizing_constant_KL(), where $\theta_1$ is the natural parameter of distribution P, and $\theta_o$ is the natural parameter of distribution Q. Here is an example of how to use it.

```python
#Prepare the data from the two target distribution
input_size = 5
num_samples= 5000
mean_p = np.ones(input_size)
mean_q = np.ones(input_size)
cov_p = np.eye(input_size)
cov_q = np.eye(input_size)
SAMPLES_P = np.random.multivariate_normal(mean_p, cov_p, num_samples) #Replace this with samples from your target distribution P 
SAMPLES_Q = np.random.multivariate_normal(mean_q, cov_q, num_samples) #Replace this with samples from your target distribution Q 
dataset = PAndQDataset(SAMPLES_P, SAMPLES_Q)

#Specify the hyperparameter and train the Neural KL estimator
hidden_size = 64
learning_rate = 0.0000075
num_epochs = 10000
batchs_size = num_samples // 2 
net,losses = train(input_size=input_size, hidden_size=hidden_size, learning_rate=0.0000075, num_epochs=num_epochs, batch_size=batchs_size, dataset=dataset)
with torch.no_grad():
  KL_estimate, _, _ = KL_DV(SAMPLES_P_tensor, SAMPLES_Q_tensor, net)

#Estiamte the log ratio of normalizing constants
parameter1 = MVN_Exponential_Family_Natural_Parameter(mean_p, cov_p) #Replace this with the natural parameter associated with exponential family distribution P 
parameter0 = MVN_Exponential_Family_Natural_Parameter(mean_q, cov_q) #Replace this with the natural parameter associated with exponential family distribution Q 
mean_sufficient_stat_1 = np.mean([MVN_Exponential_Family_Sufficient_Statistics(x) for x in SAMPLES_P],axis = 0) #Replace this with the sample mean sufficient statistics of P
Delta_estimate = log_ratio_normalizing_constant_KL(parameter1, parameter0,mean_sufficient_stat_1, KL_estimate) 
