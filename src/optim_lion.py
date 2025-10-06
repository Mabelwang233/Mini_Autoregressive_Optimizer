from torch.optim.optimizer import Optimizer

class Lion(Optimizer):
    """Minimal Lion optimizer."""
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def sign(x): return x.sign()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = p.detach().new_zeros(p.shape)
                m = state['exp_avg']
                if wd != 0:
                    p.data.add_(p.data, alpha=-wd*lr)        # decoupled weight decay
                m.data.mul_(beta1).add_(g, alpha=1-beta1)
                p.data.add_(self.sign(m), alpha=-lr)
                m.data.mul_(beta2).add_(g, alpha=1-beta2)
                p.data.add_(self.sign(m), alpha=-lr)
        return loss
