import torch
import torch.distributions as dist


class HMMPoisson(torch.nn.Module):
    """
    Hidden Markov Model with discrete observations.
    """
    def __init__(self, n_states, m_dimensions, max_itterations = 100, tolerance = 0.1, verbose = True, use_cuda = True):
        super(HMMPoisson, self).__init__()
        self.n_states = n_states  # number of states
        self.T_max = None # Max time step
        self.m_dimensions = m_dimensions
        
        self.max_itterations = max_itterations
        self.tolerance = tolerance
        
        self.verbose = verbose
        
        # A_mat
        self.transition_matrix = torch.nn.functional.softmax(torch.rand(self.n_states, self.n_states)*10, dim = 0)
        self.log_transition_matrix = self.transition_matrix.log()
        
        # b(x)
        self.lambdas = torch.exp(torch.rand(self.n_states, m_dimensions)*10)
        self.log_emission_matrix = None
        self.emission_matrix = None

        # pi
        self.unnormalized_state_priors = torch.nn.Parameter(torch.randn(self.n_states))
        self.log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
        self.state_priors = torch.exp(self.log_state_priors)
        

        # use the GPU
        self.is_cuda = (torch.cuda.is_available() and use_cuda)
        if self.is_cuda:
            self.cuda()

    def emission_model(self, x, log = True):
        """Calculate log-probability of each observation for each state
        x: LongTensor of shape (T_max, m_dimensions)

        Get observation log probabilities
        """
        # Compute Poisson log probabilities for each lambda in parallel
        poisson_dist = dist.Poisson(self.lambdas)
        log_probabilities = torch.zeros(x.shape[0], self.n_states)
        for t in range(x.shape[0]):
            log_probabilities[t,:] = poisson_dist.log_prob(x[t,:]).sum(dim=1)
        if log:
            return log_probabilities
        
        return log_probabilities.exp()
        
                    
    def log_alpha_calc(self):
        """
        self.log_emission_matrix: longTensor of shape (T_max, n_states)

        Returns:
            log_alpha: floatTensor of shape (T_max, n_states)
        """
        assert self.log_emission_matrix is not None, "No emission matrix"
        assert self.T_max is not None, "No maximum time"
        
        log_alpha = torch.zeros(self.T_max, self.n_states).float()
        if self.is_cuda:
            log_alpha = log_alpha.cuda()

        log_alpha[0, :] = self.log_emission_matrix[0, :] + self.log_state_priors
        
        
        # log_alpha[1:self.T_max,:] = self.log_emission_matrix[1:self.T_max,:] + self.log_transition_matrix[0:(self.T_max-1), :]

        for t in range(1, self.T_max):
            log_alpha[t, :] = self.log_emission_matrix[t, :] + log_domain_matmul(self.log_transition_matrix, log_alpha[t-1, :])
        
        return log_alpha
    
    def log_beta_calc(self):
        assert self.log_emission_matrix is not None, "No emission matrix"
        
        log_beta = torch.zeros(self.T_max, self.n_states).float()
        if self.is_cuda:
            log_beta = log_beta.cuda()
        
        
        for t in range(self.T_max - 2, -1, -1):
            beta_t_s = torch.zeros(self.n_states).float()
            for s in range(self.n_states):
                log_probs = log_beta[t + 1,] + self.log_transition_matrix[:, s] + self.log_emission_matrix[t + 1, s]
                beta_t_s = torch.logsumexp(torch.stack([beta_t_s, log_probs], dim=0), dim=0)
            log_beta[t,:] = beta_t_s
        
        return log_beta
        
    
    def forward(self, x):
        """ Calculate log-likelihood
        x: IntTensor of shape (T_max, m_dimensions)

        Compute log p(x)
        """
        self.T_max = x.shape[0]
        self.log_emission_matrix = self.emission_model(x)
        log_alpha = self.log_alpha_calc()

        log_prob = log_alpha[self.T_max-1, :].logsumexp(dim=0)
        return log_prob
    
    def get_lambdas(self):
        return self.lambdas
            
    
    def fit(self, x):
        """ Estimates optimal transition matrix and lambdas given the data x. Baun-Welch Algorithm

        Args:
            x (torch): T_max x m_dimensions
            log_alpha (torch) : T_max x N
            log_beta (torch) : T_max x N
        """
        
        self.T_max = x.shape[0]
        prev_log_likelihood = float('-inf')
        log_x = torch.log(x + 1e-16)
        
        for iteration in range(self.max_itterations):
            # Get emission matrix
            self.log_emission_matrix = self.emission_model(x)
            
            # E step
            ## Calculate log_alpha
            log_alpha = self.log_alpha_calc()
            
            ## Caculcate log_beta
            log_beta = self.log_beta_calc()
            
            # Chack for tolerance
            log_likelihood = log_alpha[self.T_max - 1, :].logsumexp(dim = 0)
            log_likelihood_change = log_likelihood - prev_log_likelihood
            prev_log_likelihood = log_likelihood
            if self.verbose:
                if log_likelihood_change > 0:
                    print(f"{iteration + 1} {log_likelihood:.4f}  +{log_likelihood_change}")
                else:
                    print(f"{iteration + 1} {log_likelihood:.4f}  {log_likelihood_change}")
            
            if log_likelihood_change < self.tolerance and log_likelihood_change > 0:
                if self.verbose:
                    print("Converged (change in log likelihood within tolerance)")
                break
            
            ## Calculate log_gamma
            gamma_numerator = log_alpha + log_beta
            gamma_denominator = gamma_numerator.logsumexp(dim=1, keepdim=True)
            
            log_gamma = gamma_numerator - gamma_denominator.expand_as(gamma_numerator)
            
            ## Calculate log_xi
            xi_numerator = (log_alpha[:-1, :, None] + self.log_transition_matrix[None, :, :] + log_beta[1:, None, :] + self.log_emission_matrix[1:, None, :])
            xi_denominator = xi_numerator.logsumexp(dim = (1,2), keepdim=True)
            
            log_xi = xi_numerator - xi_denominator
            
            # M step
            ## Update pi
            self.log_state_priors = log_gamma[0,] - log_gamma.logsumexp(dim = 0)
            
            ## Updaten transition matrix
            trans_numerator = log_xi.logsumexp(dim = 0)
            trans_denominator = log_gamma[0:(self.T_max-1),:].logsumexp(dim = 0)           

            self.log_transition_matrix = trans_numerator - trans_denominator.view(-1, 1)
            
            ## Update lambda
            lambda_numerator = log_domain_matmul(log_gamma.t(), log_x, dim_1=False)
            lambda_denominator = log_gamma.logsumexp(dim = 0)
            
            self.lambdas = torch.exp(lambda_numerator - lambda_denominator.view(-1,1))
            
            
            if self.verbose and iteration == self.max_itterations -1:
                print("Max itteration reached.")
                
    
    def predict(self, x):
        """
        x: IntTensor of shape (T_max, m_dimensions)

        Find argmax_z log p(z|x)
        """
        if self.is_cuda:
            x = x.cuda()

        T_max = x.shape[0]
        log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
        log_delta = torch.zeros(T_max, self.n_states).float()
        psi = torch.zeros(T_max, self.n_states).long()
        if self.is_cuda:
            log_delta = log_delta.cuda()
            psi = psi.cuda()

        self.log_emission_matrix = self.emission_model(x)
        
        log_delta[0, :] = self.log_emission_matrix[0,:] + log_state_priors
        for t in range(1, T_max):
            max_val, argmax_val = log_domain_matmul(self.log_transition_matrix, log_delta[t-1,:], max = True)
            log_delta[t, :] = self.log_emission_matrix[t,:] + max_val
            psi[t, :] = argmax_val

        # This next part is a bit tricky to parallelize across the batch,
        # so we will do it separately for each example.
        z_star = torch.zeros(T_max).long()
        z_star[T_max-1] = log_delta[T_max-1, :].argmax()
        for t in range(T_max-2, -1, -1):
            z_star[t] = psi[t+1, z_star[t+1]]

        return z_star

    def get_transition_matrix(self):
        return torch.exp(self.log_transition_matrix)


def log_matrix_multiply(log_A, log_B):
    # Ensure that the dimensions match for element-wise addition
    assert log_A.shape[1] == log_B.shape[0], "Inner dimensions do not match for matrix multiplication"

    # Perform element-wise addition in log-space
    log_result = log_A.unsqueeze(2) + log_B.unsqueeze(0)

    # Calculate the log of the sum of exponentiated values (equivalent to log-domain matrix multiplication)
    log_result = torch.logsumexp(log_result, dim=1)

    return log_result


def log_domain_matmul(log_A, log_B, dim_1 = True, max = False):
    """
    log_A: m x p
    log_B: n x p
    output: m x p matrix

    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
    """
    if not dim_1:
        m = log_A.shape[0] 
        n = log_A.shape[1]
        p = log_B.shape[1]

        log_A = torch.stack([log_A] * p, dim=2)
        log_B = torch.stack([log_B] * m, dim=0)

    elementwise_sum = log_A + log_B
    if max:
        out1, out2 = torch.max(elementwise_sum, dim = 1)
        return out1, out2
    
    out = torch.logsumexp(elementwise_sum, dim=1)
    return out

def maxmul(log_A, log_B):
    elementwise_sum = log_A + log_B
    out1, out2 = torch.max(elementwise_sum, dim=1)

    return out1, out2
