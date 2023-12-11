import torch

from utils.nf.affine_constant_flow import AffineConstantFlow

class ActNorm(AffineConstantFlow):
    """
    	Really an AffineConstantFlow but with a data-dependent initialization,
	    where on the very first batch we clever initialize the s,t so that the output
	    is unit gaussian. As described in Glow paper.
		
		Refs:
    	- https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py
    	- Glow: Generative flow with invertible 1×1 convolutions
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False
    
    def forward(self, x):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(x)