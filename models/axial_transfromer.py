from axial_attention import AxialAttention
import torch.nn as nn
class AxialTemporalTransformer(nn.Module):
    def __init__(self, ch=128, num_frames=5):
        super(AxialTemporalTransformer, self).__init__()
        self.num_frames = num_frames
        self.attention = AxialAttention(dim=ch, dim_index=2, heads=4, num_dimensions=3)
    def reshape_frames(self, x, mode:int=0):
        if mode == 0:
            b, c, h, w = x.shape
            b_new = b // self.num_frames
            x = x.reshape(b_new, self.num_frames, c, h, w)
        elif mode == 1:
            b, t, c, h, w = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x = x.reshape(b*t, c, h, w)
        return x
    def forward(self, x):
        #x of shape B X C X H X W
        #reshape to B X T X C X H X W
        x = self.reshape_frames(x, 0)
        x = self.attention(x)
        x = self.reshape_frames(x, 1)
        return x