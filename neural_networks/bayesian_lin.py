from pyro.infer import Trace_ELBO
from pyro.nn import PyroSample
import torch


 class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        W = torch.randn(in, out)
        b = torch.rand(out) * torch.pi * 2
        self.register_buffer("W", W)
        self.register_buffer("b", b)

        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        t = x @ self.W + self.b
        t = torch.cos(t)
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(t).squeeze(-1)
        with pyro.plate("data", t.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean
