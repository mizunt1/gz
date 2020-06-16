import torch
def cat(a, b, dim):
    dims_a = len(a.shape)
    dims_b = len(b.shape)
    diff = abs(dims_a - dims_b)
    if dims_a == dims_b:
        # both must have dims (batch_size, height)
        return torch.cat((a, b) , dim)
    if dims_a < dims_b:
        to_be_expanded = a
        not_to_be_expanded = b
    else:
        to_be_expanded = b
        not_to_be_expanded = a
    for i in range(diff):
        to_be_expanded = to_be_expanded.unsqueeze(0)
        # add 1s to lhs of the tensor to be expanded
        expand_num = not_to_be_expanded.shape[(-1*len(to_be_expanded.shape)) - i ]
        to_be_expanded = to_be_expanded.expand(expand_num, *to_be_expanded.shape[1:])
        # how much we want to expand depends on the size of that
        # dimension for the non expanding tensor
    return torch.cat((to_be_expanded, not_to_be_expanded), dim)
        
