import torch
from quantizers.uniform import *


class WrapNet:
    def __init__(self, layer):
        self.layer = layer        
        self.quantizer = UniformQuantizer()
        self.inps = None
        self.outs = None
   

    def quant(self):
        assert self.quantizer is not None, "Quantizer should be defined first."

        W = self.layer.weight.data.clone()
        W = W.float()
        if len(W.shape) == 4:
            W = W.flatten(1, 3)
            Q = self.quantizer.forward(W)
            Q = Q.view(self.layer.weight.data.shape)    
        else:
            Q = self.quantizer.forward(W)
        
        # assign quantized (fake-quant) weights
        self.layer.weight.data = Q
    

    def free(self):
        self.inps = None
        self.outs = None

        torch.cuda.empty_cache()