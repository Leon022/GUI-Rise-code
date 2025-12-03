import pdb
import torch

def get_select_mask(tensor, skip_ratio=0, rand=False):
    # Use tensor operations for efficiency
    retain_mask = (tensor == -1).clone()
    unique_vals, counts = torch.unique(tensor, return_counts=True)

    for i, (val, count) in enumerate(zip(unique_vals, counts)):
        if val == -1:
            continue
        positions = (tensor == val).nonzero(as_tuple=True)[0]
        num_positions = len(positions)
        
        if num_positions == 1:
            retain_mask[positions] = True
        else:
            num_to_skip = int(round(num_positions * skip_ratio))
            num_to_retain = max(1, num_positions - num_to_skip)
            if rand:
                # rand means random select subset of selective tokens for layer-wise
                perm = torch.randperm(num_positions, device=tensor.device)
                positions_to_retain = positions[perm[:num_to_retain]]
            else:
                indices = torch.linspace(0, num_positions - 1, steps=num_to_retain).long()
                positions_to_retain = positions[indices]
                
            retain_mask[positions_to_retain] = True
    return retain_mask