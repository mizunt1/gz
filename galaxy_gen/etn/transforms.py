"""
This defines standalone transform objects, which allow conversion from the group parameters
to an appropriate sampling grid.
"""

import torch
import torch.nn.functional as F



def broadcasting_grid_sample(
    x, grid, align_corners=False, mode="bilinear", padding_mode="zeros"
):
    batch_dims = x.shape[:-3]
    if not batch_dims == grid.shape[:-3]:
        raise RuntimeError("Batch dimensions of x and grid must be consistent")
    output_size = (
        x.shape[-3],
        grid.shape[-3],
        grid.shape[-2]
    )  # remove the need to manually specifiy the channel dimension
    x = torch.flatten(x, 0, -4)
    grid = torch.flatten(grid, 0, -4)

    render = F.grid_sample(
        x, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode
    )
    return render.view(*batch_dims, *output_size)



class GridTransform(object):
    def __init__(self, transform):
        """
        A grid-to-grid transformation.
        
        Args:
            transform: Callable, a tensor-to-tensor mapping where the input and output
            dimensions are both [batch, height, width, 2]
        """
        self.transform = transform
    
    def __call__(self, grid):
        """Perform the transformation on a grid.
        
        Args:
            grid: torch.Tensor, tensor of shape [batch, height, width, 2] denoting
                the (x, y) coordinates of each of the height x width grid points
                for each grid in the batch.
                
        Returns:
            A tensor of shape [batch, height, width, 2] representing the transformed grid.
        """
        return self.transform(grid)
    
    def compose(self, other):
        """Compose with another transformation"""
        if not isinstance(other, GridTransform):
            raise ValueError('Invalid type')
        return GridTransform(lambda x: self(other(x)))

    
class ProjectiveGridTransform(GridTransform):
    def __init__(self, transform):
        """
        A grid-to-grid projective transformation.
        
        Args:
            transform: torch.Tensor, a tensor with dimensions [batch, 3, 3] representing a collection
            of projective transformations.
        """
        super().__init__(transform)
    
    def __call__old_(self, grid):
        """Perform the transformation on a grid.
        
        Args:
            grid: torch.Tensor, tensor of shape [batch, height, width, 2] denoting
                the (x, y) coordinates of each of the height x width grid points
                for each grid in the batch.
                
        Returns:
            A tensor of shape [batch, height, width, 2] representing the transformed grid.
        """
        n, h, w, _ = grid.shape
        ones = grid.new_ones(n, h, w, 1)
        coords = torch.cat([grid, ones], -1)
        coords = torch.bmm(coords.view(n, h*w, 3), self.transform.permute(0, 2, 1))
        coords = coords.view(n, h, w, 3)
        grid_tf = torch.empty_like(grid)
        grid_tf[:, :, :, 0] = coords[:, :, :, 0].div(coords[:, :, :, 2] + 1e-8)
        grid_tf[:, :, :, 1] = coords[:, :, :, 1].div(coords[:, :, :, 2] + 1e-8)
        return grid_tf

    def __call__(self, grid):
        """Perform the transformation on a grid.
        
        Args:
            grid: torch.Tensor, tensor of shape [batch, height, width, 2] denoting
                the (x, y) coordinates of each of the height x width grid points
                for each grid in the batch.
                
        Returns:
            A tensor of shape [batch, height, width, 2] representing the transformed grid.
        """
        ones = grid.new_ones(*grid.shape[:-1], 1)
        coords = torch.cat([grid, ones], -1)
        coords = torch.squeeze(self.transform[..., None, None, :, :] @ coords[..., None], -1)
        # renormalise and drop last dimension
        coords = coords / (coords[..., 2] + 1e-8).unsqueeze(-1)
        coords = coords[..., :-1]

        return coords
    
    def compose(self, other):
        """Compose with another transformation"""
        if isinstance(other, ProjectiveGridTransform):
            new_mat = self.transform @ other.transform
            return ProjectiveGridTransform(new_mat)
        elif isinstance(other, GridTransform):
            return super().compose(other)
        else:
            raise ValueError('Invalid type')

class Transform:
    def __init__(self, periodic_v=False, periodic_u=False, has_u=True, has_v=True):
        self.has_u = has_u
        self.has_v = has_v
        self.periodic_u = periodic_u
        self.periodic_v = periodic_v

    @staticmethod
    def transform_from_params(*params):
        raise NotImplementedError

    def inverse_transform_from_params(self, params):
        return self.transform_from_params(*[-p for p in params])


class Translation(Transform):
        
    def transform_from_params(self, *params):
        tx, ty = params
        mat = torch.zeros(*tx.shape, 3, 3, device=tx.device)
        mat[..., 0, 0] = 1.
        mat[..., 1, 1] = 1.
        mat[..., 2, 2] = 1.
        mat[..., 0, 2] = tx
        mat[..., 1, 2] = ty
        return ProjectiveGridTransform(mat)
    

class Rotation(Transform):    
    def __init__(self):
        super().__init__(periodic_v=True,has_u=False)
    def transform_from_params(self, *params):
        angle = params[0]
        device = angle.device
        ca, sa = torch.cos(angle), torch.sin(angle)
        mat = torch.zeros(*angle.shape, 3, 3, device=angle.device)
        mat[..., 0, 0] =  ca
        mat[..., 0, 1] = -sa
        mat[..., 1, 0] =  sa
        mat[..., 1, 1] =  ca
        mat[..., 2, 2] =  1.      
        return ProjectiveGridTransform(mat)
    
    
class Scale(Transform):
        
    def __init__(self):
        super().__init__(periodic_v=True, has_v=False)
    def transform_from_params(self, *params):
        scale = torch.exp(params[0])
        mat = torch.zeros(*scale.shape, 3, 3, device=scale.device)
        mat[..., 0, 0] = scale
        mat[..., 1, 1] = scale
        mat[..., 2, 2] = 1.        
        return ProjectiveGridTransform(mat)

    
class RotationScale(Transform):    
        
    def __init__(self):
        super().__init__(periodic_v=True)

    def transform_from_params(self, *params):
        scale, angle = params
        scale = torch.exp(scale)

        device = scale.device
        ca, sa = torch.cos(angle), torch.sin(angle)
        mat = torch.zeros(*scale.shape, 3, 3, device=device)
        mat[..., 0, 0] =  scale * ca
        mat[..., 0, 1] = -scale * sa
        mat[..., 1, 0] =  scale * sa
        mat[..., 1, 1] =  scale * ca
        mat[..., 2, 2] = 1.        
        return ProjectiveGridTransform(mat)
        
    
class ShearX(Transform):
        
    def __init__(self):
        super().__init__(has_u=False)

    def transform_from_params(self, *params):
        shear = params[0]
        mat = torch.zeros(*shear.shape, 3, 3, device=shear.device)
        mat[..., 0, 0] = 1.
        mat[..., 0, 1] = shear
        mat[..., 1, 1] = 1.
        mat[..., 2, 2] = 1.        
        return ProjectiveGridTransform(mat)
    
    
class ShearY(Transform):
    
    def __init__(self):
        super().__init__(has_u=False)

    def transform_from_params(self, *params):
        shear = params[0]
        mat = torch.zeros(*shear.shape, 3, 3, device=shear.device)
        mat[..., 0, 0] = 1.
        mat[..., 1, 0] = shear
        mat[..., 1, 1] = 1.
        mat[..., 2, 2] = 1.
        return ProjectiveGridTransform(mat)
    
    
class ScaleX(Transform):    
    
    def __init__(self):
        super().__init__(has_v=False)

    def transform_from_params(self, *params):
        scale = torch.exp(params[0])
        mat = torch.zeros(*scale, 3, 3, device=scale.device)
        mat[..., 0, 0] = scale
        mat[..., 1, 1] = 1.
        mat[..., 2, 2] = 1.       
        return ProjectiveGridTransform(mat)
        
        
class ScaleY(Transform):    
    
    def __init__(self):
        super().__init__(has_v=False)

    def transform_from_params(self, *params):
        scale = torch.exp(params[0])
        mat = torch.zeros(*scale.shape, 3, 3, device=scale.device)
        mat[..., 0, 0] = 1.
        mat[..., 1, 1] = scale
        mat[..., 2, 2] = 1.       
        return ProjectiveGridTransform(mat)
    
    
class HyperbolicRotation(Transform):
        
    def __init__(self):
        super().__init__(has_u=False)

    def transform_from_params(self, *params):
        scale = torch.exp(params[0])
        mat = torch.zeros(*scale.shape, 3, 3, device=scale.device)
        mat[..., 0, 0] = scale
        mat[..., 1, 1] = 1./scale
        mat[..., 2, 2] = 1.
        return ProjectiveGridTransform(mat)
    
    
class PerspectiveX(Transform):
               
    def __init__(self):
        super().__init__(has_v=False)
    def transform_from_params(self, *params):
        perspective = params[0]
        mat = torch.zeros(*perspective.shape, 3, 3, device=perspective.device)
        mat[..., 0, 0] = 1.
        mat[..., 1, 1] = 1.
        mat[..., 2, 0] = perspective
        mat[..., 2, 2] = 1.
        return ProjectiveGridTransform(mat)
    
    
class PerspectiveY(Transform):
               
    def __init__(self):
        super().__init__(has_v=False)

    def transform_from_params(self, *params):
        perspective = params[0]
        mat = torch.zeros(*perspective.shape, 3, 3, device=perspective.device)
        mat[..., 0, 0] = 1.
        mat[..., 1, 1] = 1.
        mat[..., 2, 1] = perspective
        mat[..., 2, 2] = 1.
        return ProjectiveGridTransform(mat)
        


class TransformSequence:
    def __init__(self, *transforms):
        """
        Container utility, containing lazily composed transforms
        and their inverses
        """
        super().__init__()
        self.transforms = transforms


    def transform_from_params(self, params):
        transform = None
        for t, p in zip(self.transforms, params):
            tr = t.transform_from_params(*p)
            if transform is not None:
                transform = transform.compose(tr)
            else:
                transform = tr
        return transform

    def inverse_transform_from_params(self, params):
        transform = None
        for t, p in reversed(list(zip(self.transforms, params))):
            tr = t.inverse_transform_from_params(p)
            if transform is not None:
                transform = transform.compose(tr)
            else:
                transform = tr
        return transform

    def __iter__(self):
        return iter(self.transforms)
