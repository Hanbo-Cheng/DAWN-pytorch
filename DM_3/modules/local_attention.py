import sys
# sys.path.append('your/path/DAWN-pytorch')
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from einops_exts import rearrange_many
import math
import concurrent.futures
# from local_attn_cuda_pkg.test_cuda import attn_forward
# from local_attn_cuda_pkg.test_cuda import attn_forward, compute_res_forward
# def attn_forward(x, y, batch_size, hw, seq_len, k_size, head, d, device):
#     attn = torch.zeros(batch_size, hw, seq_len, k_size, head, device=device)
#     local_attn_res.attn_cuda(x, y, attn, batch_size, hw, seq_len, k_size, head, d)
#     return attn

# def compute_res_forward(attn, z, batch_size, hw, seq_len, head, head_dim, k_size, device):
#     res = torch.zeros(batch_size, hw, seq_len, head, head_dim, device=device)
#     local_attn_res.compute_res_cuda(attn, z, res, batch_size, hw, seq_len, head, head_dim, k_size)
#     return res

def exists(x):
    return x is not None

def to_mask(x, mask, mode='mul'):
    if mask is None:
        return x
    else:
        while x.dim() > mask.dim():
            mask = mask.unsqueeze(-1)
        if mode == 'mul':
            return x * mask
        else:
            return x + mask

# def extract_seq_patches(x, kernel_size, rate):
#     """x.shape = [batch_size, seq_len, seq_dim]"""
#     seq_len = x.size(1)
#     seq_dim = x.size(2)
#     k_size = kernel_size + (rate - 1) * (kernel_size - 1)
#     p_right = (k_size - 1) // 2
#     p_left = k_size - 1 - p_right
#     x = F.pad(x, (0, 0, p_left, p_right), mode='constant', value=0)
#     xs = [x[:, i: i + seq_len] for i in range(0, k_size, rate)]
#     x = torch.cat(xs, dim=2)
#     return x.reshape(-1, seq_len, kernel_size, seq_dim)

def extract_seq_patches(x, kernel_size, rate):
    """x.shape = [batch_size, hw, seq_len, seq_dim]"""
    # batch_size, hw, seq_len, seq_dim = x.size()

    # Calculate the size of the expanded kernel and the number of padding to be added on both sides.
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right

    # padding
    x = F.pad(x, (0, 0, p_left, p_right), mode='constant', value=0)  # pad only the second dimension

    # Use the unfold method to extract sliding windows.
    x_unfold = x.unfold(dimension=2, size=k_size, step=rate)  # x, window, k_size, step, rate
    x_unfold = x_unfold.transpose(-1, -2)
    
    #  reshape (batch_size, hw, seq_len, kernel_size, seq_dim)
    x_patches = x_unfold[:, :, :, ::rate]

    return x_patches

def window_attn(x, y, z, kernel_size, mask, rate):
    """y.shape x.shape = [batch_size, hw, seq_len, self.heads, dim_head]"""
    batch_size, hw, seq_len, head, head_dim = x.size()
    device = x.device

    # Calculate the size of the expanded kernel and the number of padding to be added on both sides.
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right

    # padding
    y = F.pad(y, (0, 0, 0, 0, p_left, p_right), mode='constant', value=0)  # pad only the second dimension
    z = F.pad(z, (0, 0, 0, 0, p_left, p_right), mode='constant', value=0)

    attn = torch.zeros(batch_size, hw, seq_len, k_size, head).to(device)
    for i in range(seq_len):
        # torch.matmul(x[:,:,i].unsqueeze(2), y[:,:,i:i + k_size].transpose()) # b, hw, 1, d ; b, hw, w, d
        attn[:,:, i] = torch.einsum('b n h d, b n w h d -> b n w h', x[:,:,i], y[:,:,i:i + k_size]) 
    # reshape (batch_size, hw, seq_len, kernel_size, seq_dim)
    # res = rearrange(res, 'b n l w h -> b n h l w')
    attn = to_mask(attn, mask.unsqueeze(0), 'add')
    attn = attn - attn.amax(dim=-2, keepdim=True).detach()
    attn = F.softmax(attn, dim=-2)
    res = torch.zeros(batch_size, hw, seq_len, head, head_dim).to(device)

    for i in range(seq_len):
        res[:,:,i] = torch.einsum('b n w h, b n w h d -> b n h d', attn[:,:,i], z[:,:,i : i +k_size])  # attn[:,:,i] * z[:,:,i : i +k_size]
    res = res.view(batch_size, hw, seq_len, -1)
    return res


def window_attn_2(x, y, z, kernel_size, mask, rate):  # bad optimization
    """
    The optimized window_attn function eliminates two explicit for loops and utilizes tensor operations for parallel computation.
    
    param:
        x (Tensor): [batch_size, hw, seq_len, heads, dim_head]
        y (Tensor): [batch_size, hw, seq_len, heads, dim_head]
        z (Tensor): [batch_size, hw, seq_len, heads, dim_head]
        kernel_size (int): window size
        mask (Tensor)
        rate (int)
    
    return:
        Tensor: [batch_size, hw, seq_len, heads * dim_head]
    """
    batch_size, hw, seq_len, head, head_dim = x.size()

    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right

    y_padded = F.pad(y, (0, 0, 0, 0, p_left, p_right), mode='constant', value=0)  # [batch_size, hw, seq_len + p_left + p_right, heads, dim_head]
    z_padded = F.pad(z, (0, 0, 0, 0, p_left, p_right), mode='constant', value=0)  

    # y_windows  z_windows  [batch_size, hw, seq_len, k_size, heads, dim_head]
    y_windows = y_padded.as_strided(
        size=(batch_size, hw, seq_len, k_size, head, head_dim),
        stride=(
            y_padded.stride(0), 
            y_padded.stride(1), 
            y_padded.stride(2), 
            y_padded.stride(2),  
            y_padded.stride(3), 
            y_padded.stride(4)
        )
    )
    
    z_windows = z_padded.as_strided(
        size=(batch_size, hw, seq_len, k_size, head, head_dim),
        stride=(
            z_padded.stride(0), 
            z_padded.stride(1), 
            z_padded.stride(2), 
            z_padded.stride(2), 
            z_padded.stride(3), 
            z_padded.stride(4)
        )
    )

    # x: [batch_size, hw, seq_len, heads, dim_head] -> [batch_size, hw, seq_len, 1, heads, dim_head]
    x_expanded = x #.unsqueeze(3)  #  [batch_size, hw, seq_len, 1, heads, dim_head]

    attn_scores = torch.einsum('b n l h d, b n l s h d -> b n l s h', x_expanded, y_windows)  

    attn = to_mask(attn_scores, mask.unsqueeze(0), 'add')  

    attn = attn - attn.amax(dim=-2, keepdim=True).detach()  
    attn = F.softmax(attn, dim=-2)  #    Softmax on k_size

    res = (attn.unsqueeze(-1) * z_windows).sum(dim=-3)  

    res = res.view(batch_size, hw, seq_len, head * head_dim)

    return res

def window_attn_stream(x, y, z, kernel_size, mask, rate):  # bad optimization
    """y.shape x.shape = [batch_size, hw, seq_len, self.heads, dim_head]"""
    batch_size, hw, seq_len, head, head_dim = x.size()

    # Calculate the size of the expanded kernel and the number of padding to be added on both sides.
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right

    # padding
    y = F.pad(y, (0, 0, 0, 0, p_left, p_right), mode='constant', value=0)  # pad only the second dimension
    z = F.pad(z, (0, 0, 0, 0, p_left, p_right), mode='constant', value=0)

    attn = torch.zeros(batch_size, hw, seq_len, k_size, head, device=x.device)
    res = torch.zeros(batch_size, hw, seq_len, head, head_dim, device=x.device)

    streams = [torch.cuda.Stream() for _ in range(seq_len)]

    def compute_attn(i):
        with torch.cuda.stream(streams[i]):
            attn[:, :, i] = torch.einsum('b n h d, b n w h d -> b n w h', x[:, :, i], y[:, :, i:i + k_size])

    def compute_res(i):
        with torch.cuda.stream(streams[i]):
            res[:, :, i] = torch.einsum('b n w h, b n w h d -> b n h d', attn[:, :, i], z[:, :, i:i + k_size])

    for i in range(seq_len):
        compute_attn(i)

    for stream in streams:
        stream.synchronize()

    attn = to_mask(attn, mask.unsqueeze(0), 'add')
    attn = attn - attn.amax(dim=-2, keepdim=True).detach()
    attn = F.softmax(attn, dim=-2)

    for i in range(seq_len):
        compute_res(i)

    for stream in streams:
        stream.synchronize()

    res = res.view(batch_size, hw, seq_len, -1)
    return res

def create_sliding_window_mask(x, win_size, rate):
    #  mask (len, len, head)
    # assert mask.dim() == 3, "The input mask must be of shape (len, len, head)"

    k_size = win_size + (rate - 1) * (win_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right

    # padding
    x = F.pad(x, (p_left, p_right), mode='constant', value=-1e10)  # pad only the second dimension
    res = []
    for i in range(x.shape[1]):
        res.append(x[:, i , i :i +k_size])

    return torch.stack(res, dim = 1) # len k_size, head

class OurLayer(nn.Module):

    def reuse(self, layer, *args, **kwargs):
        outputs = layer(*args, **kwargs)
        return outputs
    
 
def heavy_computation(x, y, attn, k_size, i):
        attn[:,:, i] = torch.einsum('b n h d, b n w h d -> b n w h', x[:,:,i], y[:,:,i:i + k_size]) 

def heavy_computation2(res, z, attn, k_size, i):
        res[:,:,i] = torch.einsum('b n w h, b n w h d -> b n h d', attn[:,:,i], z[:,:,i : i +k_size])  # attn[:,:,i] * z[:,:,i : i +k_size]

from functools import partial
def window_attn_mp(x, y, z, kernel_size, mask, rate):
    """y.shape x.shape = [batch_size, hw, seq_len, self.heads, dim_head]"""
    batch_size, hw, seq_len, head, head_dim = x.size()
    device = x.device
    # Calculate the size of the expanded kernel and the number of padding to be added on both sides.
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right
    
    # padding
    y = F.pad(y, (0, 0, 0, 0, p_left, p_right), mode='constant', value=0)  # pad only the second dimension
    z = F.pad(z, (0, 0, 0, 0, p_left, p_right), mode='constant', value=0)

    attn =  torch.zeros(batch_size, hw, seq_len, k_size, head).to(device)

    

    unary = partial(heavy_computation, x,y,attn, k_size)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(unary, list(range(seq_len)))
    # reshape (batch_size, hw, seq_len, kernel_size, seq_dim)
    # res = rearrange(res, 'b n l w h -> b n h l w')
    attn = to_mask(attn, mask.unsqueeze(0), 'add')
    attn = attn - attn.amax(dim=-2, keepdim=True).detach()
    attn = F.softmax(attn, dim=-2)
    res = torch.zeros(batch_size, hw, seq_len, head, head_dim).to(device)

    unary2 = partial(heavy_computation2, res,z,attn, k_size)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(unary2, list(range(seq_len)))
    res = res.view(batch_size, hw, seq_len, -1)
    return res

class LocalSelfAttention_opt(OurLayer):

    def __init__(self, d_model, heads, size_per_head, neighbors=3, rate=1, rotary_emb=None,
                 key_size=None, mask_right=False):
        super(LocalSelfAttention_opt, self).__init__()
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.neighbors = neighbors
        self.rate = rate
        self.mask_right = mask_right

        self.rotary_emb = rotary_emb
        # self.q_dense = nn.Linear(self.key_size * self.heads, self.key_size * self.heads, bias=False)
        # self.k_dense = nn.Linear(self.key_size * self.heads, self.key_size * self.heads, bias=False)
        # self.v_dense = nn.Linear(self.key_size * self.heads, self.key_size * self.heads, bias=False)
        # self.q_dense.weight.data.fill_(1)
        # self.k_dense.weight.data.fill_(1)
        # self.v_dense.weight.data.fill_(1)
        self.to_qkv = nn.Linear(d_model, self.key_size * self.heads * 3, bias=False)
        self.to_out = nn.Linear(self.key_size * self.heads, d_model, bias=False)
        # self.to_qkv.weight.data.fill_(1)
        # self.to_out.weight.data.fill_(1)

    def forward(self, inputs, pos_bias,  focus_present_mask=None,):
        # if isinstance(inputs, list):
        #     x, x_mask = inputs
        # else:
        #     x, x_mask = inputs, None
        x = inputs
        x_mask = pos_bias

        kernel_size = 1 + 2 * self.neighbors

        # if x_mask is not None:
        #     xp_mask = create_sliding_window_mask(x_mask, kernel_size, self.rate) # b, hw, seq, d_model -> b, hw, seq, win, d_model

        batch_size, hw, seq_len, seq_dim = x.size()

        if x_mask is not None:
            xp_mask = x_mask.unsqueeze(0) # b, hw, seq, win, 1
            v_mask = xp_mask
        else:
            v_mask = None

        # k = self.k_dense(x)
        # v = self.v_dense(x)
        qw, k, v = self.to_qkv(x).chunk(3, dim=-1) # qw: b, hw, seq_len, d_model
        
        qw = qw/ (self.key_size ** 0.5)
        qw = qw.view(batch_size, hw, seq_len, self.heads, self.key_size)
        k = k.view(batch_size, hw, seq_len, self.heads, self.key_size) # b, hw,  seq_len,h, d_head
        v = v.view(batch_size, hw, seq_len, self.heads, self.key_size)
        st = time.time()
        if exists(self.rotary_emb):
            qw = self.rotary_emb.rotate_queries_or_keys(qw)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        ed = time.time()
        # print("rope local: ", ed - st)
        st = time.time()
        # qw = qw.view(batch_size * hw, seq_len, seq_dim) # b * hw, seq, d_model
        # k = k.view(batch_size, hw, seq_len, self.key_size * self.heads)
        
        res = window_attn(qw, k, v, kernel_size, v_mask.permute(0, 2, 3, 1), rate = 1)
        ed = time.time()
        # print("rope local: ", ed - st)
        return self.to_out(res)
    

class MultiHeadLocalAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size):
        super(MultiHeadLocalAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        # self.out = nn.Linear(d_model, d_model)

        self.query.weight.data.fill_(1)
        self.key.weight.data.fill_(1)
        self.value.weight.data.fill_(1)
        self.query.bias.data.fill_(0)
        self.key.bias.data.fill_(0)
        self.value.bias.data.fill_(0)


    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.d_model

        Q = self.split_heads(self.query(x), batch_size)
        K = self.split_heads(self.key(x), batch_size)
        V = self.split_heads(self.value(x), batch_size)

        # Create the attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.depth ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)

        # Create the mask
        mask = (torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)).abs()
        mask = (mask > self.window_size).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask = mask.to(x.device)

        # Apply the mask to the attention scores
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Compute the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # Compute the output
        output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, depth)
        output = output.permute(0, 2, 1, 3) # (batch_size, seq_len, num_heads, depth)
        output = output.reshape(batch_size, seq_len, d_model)

        return output


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

        self.to_qkv.weight.data.fill_(1)
        self.to_out.weight.data.fill_(1)

    def forward(
            self,
            x,
            pos_bias=None,
            focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        st = time.time()
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        ed = time.time()
        print("rope normal: ", ed - st)
        # similarity

        sim = torch.einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = torch.einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        # return self.to_out(out)
        return out
    
class RelativePositionBias(nn.Module):
    def __init__(
            self,
            heads=8,
            num_buckets=32,
            max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        mask = - (((rel_pos > 20) + (rel_pos < - 20)) * (1e10))
        values = self.relative_attention_bias(rp_bucket)
        return mask +  rearrange(values, 'i j h -> h i j')
    
if __name__ == "__main__":
    # Example usage:
    d_model = 256
    window_size = 20
    seq_len = 200
    batch_size = 1
    head = 4
    res_pos = RelativePositionBias(heads=4, max_distance=32)
    rope = RotaryEmbedding(min(64, d_model//head), seq_before_head_dim = True)
    rope2 = RotaryEmbedding(min(64, d_model//head))
    model = LocalSelfAttention_opt(d_model, head, d_model//head, window_size, rotary_emb=rope)

    model_2 = Attention(d_model, head, dim_head= d_model//head, rotary_emb = rope2)
    
    
    rp = res_pos(200, 'cpu')
    xp_mask = create_sliding_window_mask(rp, 2 * window_size + 1, 1)
    for i in range(5):
        x = torch.randn(batch_size, 9, seq_len, d_model)
        st = time.time()
        output = model([x, xp_mask])
        ed = time.time()
        print("optimized: ", ed - st)
        st = time.time()
        output_2 = model_2(x , pos_bias = rp)
        ed = time.time()
        print("origin: ", ed - st)
        print(((output - output_2)**2).mean())