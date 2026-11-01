import torch.nn
import torch.nn.functional as F


class Alpha(torch.nn.Module):
    EPS = 1e-9

    def __init__(self, elem_size, config):
        super(Alpha, self).__init__()
        self.elem_size = elem_size
        self.config = config
        self.alpha = torch.nn.Parameter(torch.ones(size=[self.elem_size]))
        self.is_frozen = False
        self.n_alive_idx_after_frozen = None  # only applicable after frozen

    def get_softmax_value(self):
        return F.softmax(self.alpha, dim=0)

    def get_alive_idx(self):
        return self.n_alive_idx_after_frozen

    def get_desc(self):
        return '{}, ent: {}'.format(
            '\t'.join(['{:.3f}'.format(elem) if elem > self.EPS else '-----' for elem in self.get_softmax_value().tolist()]),
            self.get_entropy()
        )

    def get_entropy(self):
        prob = self.get_softmax_value()
        # Add epsilon to prevent log(0) and ensure numerical stability
        prob_safe = torch.clamp(prob, min=self.EPS)
        ent = torch.sum(-prob * torch.log(prob_safe))
        return ent

    def hard_threshold_and_freeze_alpha(self, num_alive_elem):
        self.is_frozen = True
        _, self.n_alive_idx_after_frozen = torch.topk(self.alpha.detach(), num_alive_elem)

        # Create new alpha values without in-place operations
        with torch.no_grad():
            new_alpha = torch.ones(self.elem_size, device=self.alpha.device, dtype=self.alpha.dtype)
            for idx in range(self.elem_size):
                if idx not in self.n_alive_idx_after_frozen:
                    new_alpha[idx] = self.EPS
                else:
                    new_alpha[idx] = 100.0
            # Replace the parameter without in-place operation
            self.alpha.data.copy_(new_alpha)

        return self.alpha
