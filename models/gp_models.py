import gpytorch, torch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import WhitenedVariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution, GridInterpolationVariationalStrategy
import pdb

# GP Layer

class GPRegressionLayer1(AbstractVariationalGP):
    def __init__(self,num_dims = 90, num_inducing_points = 40):
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points = num_inducing_points,batch_size = num_dims)
        inducing_points = torch.rand(num_dims,num_inducing_points,1)        
        variational_strategy = WhitenedVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPRegressionLayer1, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_size=num_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_size=num_dims), batch_size=num_dims
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)










