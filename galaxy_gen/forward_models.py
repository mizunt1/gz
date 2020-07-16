import torch
from torch import nn
from torch import functional as F
import pyro
from pyro import distributions as D
from galaxy_gen.etn import transforms
from torch.distributions import constraints
import typing as T
from galaxy_gen.etn.transforms import ProjectiveGridTransform

def random_pose_transform(
    transforms: transforms.TransformSequence, device=torch.device("cpu")
):
    """
    Take a sequence of transformations and create a r.v corresponding to the co-ordinates of each.
    Return the transformation corresponding to this that can be applied to an input image.
    TODO: atm these classes have a cnn attached; this doesn't feel like a super clean abstraciton, but we'll see...
    """
    params = []

    for transform in transforms:
        # sample the U and V corresponding to that transformation
        sample_site_prefix = repr(type(transform)).split(".")[-1].strip("'>")

        u_sample_name = sample_site_prefix + "_u"
        v_sample_name = sample_site_prefix + "_v"

        # todo: make this more sensible
        ps = []
        if transform.has_u:
            if transform.periodic_u:
                u = pyro.sample(
                    u_sample_name,
                    D.VonMises(
                        torch.tensor(0.0, device=device),
                        torch.tensor(1e-2, device=device),
                    ),
                )
            else:
                u = pyro.sample(
                    u_sample_name,
                    D.Normal(
                        torch.tensor(0.0, device=device),
                        torch.tensor(1.0, device=device),
                    ),
                )
            ps.append(u)

        if transform.has_v:
            if transform.periodic_v:
                v = pyro.sample(
                    v_sample_name,
                    D.VonMises(
                        torch.tensor(0.0, device=device),
                        torch.tensor(1e-2, device=device),
                    ),
                )
            else:
                v = pyro.sample(
                    v_sample_name,
                    D.Normal(
                        torch.tensor(0.0, device=device),
                        torch.tensor(1.0, device=device),
                    ),
                )
            ps.append(v)

        params.append(ps)

    # we take the inverse transformation here because we want to use the order specified by
    # the spatial transformer network, which uses the opposite convection to what make sense
    # from a generative perspective
    if len(params) > 0:
        transform_grid = transforms.inverse_transform_from_params(params)
    else:
        # handle the edge case where we have an empty list of transformations.
        transform_grid = ProjectiveGridTransform(torch.eye(3,device=device))
    return transform_grid


def transforming_template_mnist(
    data, label, transforms, cond=True, obs_label=True, grid_size=28
):
    with pyro.plate(data.shape[0]):
        # firsnt attempt - literally try to learn a template to check
        # we got the tranformation logic correct
        template = pyro.sample(
            torch.rand(1, 32, 32), constraint=constraints.unit_interval
        )

        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size)
            ),
            -1,
        )

        transform = random_pose_transform(transforms)

        transform_grid = transform(grid)

        transformed_template = T.broadcasting_grid_sample(
            template.expand(data.shape[0], 1, 32, 32), transform_grid
        )

        pyro.sample("pixels", D.Bernoulli(transformed_template).to_event(3))
