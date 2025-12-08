import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn.init import kaiming_uniform_, constant_
# from models.timesnet import TimesBlock

class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, gates = torch.split(x, x.shape[-1] // 2, dim=-1)
        return x * torch.nn.functional.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dense1 = nn.Linear(dim, dim * mult * 2)
        self.activation = GEGLU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(dim * mult, dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.heads = heads

    def forward(self, x):
        h = self.heads
        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = torch.split(qkv, qkv.shape[-1] // 3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = torch.nn.functional.softmax(sim, dim=-1)
        dropped_attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)

        return out, attn

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                           FeedForward(dim=dim, dropout=ff_dropout)])
            for _ in range(depth)
        ])

    def forward(self, x, return_attn=False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)
            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x
        return x, torch.stack(post_softmax_attns)

class FTTransformer(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        dim = args.dim
        depth = 6
        heads = args.heads
        attn_dropout = args.attn_dropout
        ff_dropout = args.ff_dropout
        self.use_cls_token = args.use_cls_token
        num_special_tokens = args.num_special_tokens
        categories = args.categories
        out = args.out
        self.out = out
        self.num_cont = args.num_cont
        num_cont = args.num_cont
        num_cate = args.num_cate

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        total_tokens = self.num_unique_categories + num_special_tokens

        if self.num_unique_categories > 0:
            categories_offset = torch.nn.functional.pad(torch.tensor(categories), (1, 0), value=num_special_tokens)
            self.categories_offset = categories_offset.cumsum(-1)[:-1].to(device)
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        if num_cont > 0:
            self.numerical_embedder = NumericalEmbedder(dim, num_cont)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.seq = args.num_cont + args.num_cate + int(args.use_cls_token)
        # self.timesnet = TimesBlock(self.seq, self.seq//2, self.seq//2, top_k=5,
        #                               d_model=dim, d_ff=dim, num_kernels=6).to(device)
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=16,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, out)
        )

    def model_name(self):
        return 'FT-Transfomer'

    def forward(self, x_categ, x_numer, label, step=None, return_attn=False):
        xs = []
        self.num_unique_categories = 0
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)
        if self.num_cont> 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)

        x = torch.cat(xs, dim=1)
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x = self.timesnet(x)
        x, attns = self.transformer(x, return_attn=True)
        x = x[:, 0]
        logits = self.to_logits(x)

        if self.out == 1:
            loss = nn.MSELoss()(logits.reshape(-1), label.float())
        else:
            loss = nn.CrossEntropyLoss()(logits, label)

        return logits, loss
