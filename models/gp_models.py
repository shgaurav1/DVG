import gpytorch, torch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import WhitenedVariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution, GridInterpolationVariationalStrategy
import pdb

# GP Layer

class GPModel(AbstractVariationalGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = WhitenedVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=16
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=16, rank=1
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)



class GPLayer(AbstractVariationalGP):
    def __init__(self, grid_size=32, grid_bounds=[(-1, 1), (-1, 1)]):
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=grid_size*grid_size)
        variational_strategy = GridInterpolationVariationalStrategy(self,
                                                                    grid_size=grid_size,
                                                                    grid_bounds=grid_bounds,
                                                                    variational_distribution=variational_distribution)
        super(GPLayer, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=16
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=16, rank=1
        )

    def forward(self, x):
        pdb.set_trace()
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class GPRegressionLayer(AbstractVariationalGP):
    def __init__(self):
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points = 40,batch_size = 16)
        inducing_points = torch.rand(16,40,1)        
        variational_strategy = WhitenedVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPRegressionLayer, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_size=16)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_size=16), batch_size=16
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        # pdb.set_trace()
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class GPRegressionLayer1(AbstractVariationalGP):
    def __init__(self):
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points = 40,batch_size = 90)
        inducing_points = torch.rand(90,40,1)        
        variational_strategy = WhitenedVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPRegressionLayer1, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_size=90)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_size=90), batch_size=90
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        # pdb.set_trace()
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)






class GPRegressionLayer2(AbstractVariationalGP):
    def __init__(self):
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points = 80,batch_size = 96)
        inducing_points = torch.rand(96,80,1)        
        variational_strategy = WhitenedVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPRegressionLayer2, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_size=96)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_size=96), batch_size=96
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        # pdb.set_trace()
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# class GPRegressionLayer3(gpytorch.models.ApproximateGP):
#     def __init__(self):
#         variational_distribution = CholeskyVariationalDistribution(num_inducing_points = 80,batch_shape = torch.Size([96]))
#         inducing_points = torch.rand(96,80,1)        
#         variational_strategy =  gpytorch.variational.MultitaskVariationalStrategy(WhitenedVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True),num_tasks = 96)
#         super(GPRegressionLayer3, self).__init__(variational_strategy)
#         self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([96]))
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(batch_shape=torch.Size([96])), batch_shape=torch.Size([96])
#         )
        
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         # pdb.set_trace()
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)







class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=16
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=16, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class MultitaskScalableGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskScalableGPModel, self).__init__(train_x, train_y, likelihood)
        
        # SKI requires a grid size hyperparameter. This util can help with that
        grid_size = 16#gpytorch.utils.grid.choose_grid_size(train_x)
        
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=16
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.RBFKernel(), grid_size=grid_size, num_dims=16,
            ), num_tasks=16, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# class MultitaskGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood, n_devices = n_devices):
#         super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.MultitaskMean(
#             gpytorch.means.ConstantMean(), num_tasks=16
#         )
#         covar_module_base = gpytorch.kernels.MultitaskKernel(
#             gpytorch.kernels.RBFKernel(), num_tasks=16, rank=1
#         )
#         self.covar_module = gpytorch.kernels.MultiDeviceKernel(
#             covar_module_base, device_ids=range(n_devices),
#             output_device=output_device
#         )

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
# class MultitaskGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

#         # SKI requires a grid size hyperparameter. This util can help with that
#         grid_size = 64#gpytorch.utils.grid.choose_grid_size(train_x)

#         self.mean_module = gpytorch.means.MultitaskMean(
#             gpytorch.means.ConstantMean(), num_tasks=128
#         )
#         self.covar_module = gpytorch.kernels.MultitaskKernel(
#             gpytorch.kernels.GridInterpolationKernel(
#                 gpytorch.kernels.RBFKernel(), grid_size=grid_size, num_dims=128,
#             ), num_tasks=128, rank=1
#         )

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)