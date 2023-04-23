import copy
import torch
from torch import nn

class Layer(nn.Module):

    def __init__(self, config):

        super(Layer, self).__init__()

        d_model = config['feature_dim']
        nhead =  config['n_head']

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source):

        bs = x.size(0)
        q, k, v = x, source, source

        qw = self.q_proj(q)
        kw = self.k_proj(k)
        vw = self.v_proj(v)

        qw = qw.view(bs, -1, self.nhead, self.dim)
        kw = kw.view(bs, -1, self.nhead, self.dim)
        vw = vw.view(bs, -1, self.nhead, self.dim)


        # attention
        a = torch.einsum("nlhd,nshd->nlsh", qw, kw)
        a = a / qw.size(3) **0.5
        a = torch.softmax(a, dim=2)
        o = torch.einsum("nlsh,nshd->nlhd", a, vw).contiguous()  # [N, L, (H, D)]

        message = self.merge(o.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        e = x + message

        return e

class Transformer(nn.Module):

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.d_model = config['feature_dim']
        self.nhead = config['n_head']
        self.layer_types = config['layer_types']

        encoder_layer = Layer(config)

        self.layers = nn.ModuleList()

        for l_type in self.layer_types:

            if l_type in ['self','cross']:

                self.layers.append( copy.deepcopy(encoder_layer))

        self._reset_parameters()



    def forward(self, src_feat, tgt_feat, timers = None):

        self.timers = timers

        assert self.d_model == src_feat.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_types) :

            if name == 'self':
                if self.timers: self.timers.tic('self atten')
                src_feat = layer(src_feat, src_feat)
                tgt_feat = layer(tgt_feat, tgt_feat)
                if self.timers: self.timers.toc('self atten')

            elif name == 'cross':
                if self.timers: self.timers.tic('cross atten')
                src_feat = layer(src_feat, tgt_feat)
                tgt_feat = layer(tgt_feat, src_feat)
                if self.timers: self.timers.toc('cross atten')
            else:
                raise KeyError

        return src_feat, tgt_feat


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)