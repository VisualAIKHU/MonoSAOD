import torch
import torch.nn.functional as F

class TwoStageBin:
    def __init__(self, dim, K=8, m=32, alpha=0.1, dtype=torch.float16, device='cuda'):
        self.dim = dim
        self.K = K
        self.m = m
        self.alpha = alpha
        self.dtype = dtype
        self.device = torch.device(device)
        self.eps = 1e-12

        self.M = torch.empty(0, dim, dtype=torch.float32, device=self.device)   # protos (float32 for stability)
        self.count = torch.empty(0, dtype=torch.int32, device=self.device)
        self.members = []  # list of tensors, each [<=m, dim] in dtype

    @torch.no_grad()
    def _add_member(self, j, f32):
        mem = self.members[j]
        if mem.size(0) < self.m:
            self.members[j] = torch.cat([mem, f32[None]], dim=0)
        else:
            # FIFO: roll and overwrite last row
            self.members[j] = torch.roll(mem, shifts=-1, dims=0)
            self.members[j][-1].copy_(f32)




