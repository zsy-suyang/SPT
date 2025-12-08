import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from einops import rearrange, repeat
from models.Embed import DataEmbedding
from models.specnet import SpecBlock
import torch.nn.init as init 

class Prediction(nn.Module):
    def __init__(self, input_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Prediction, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_hidden_1),
            nn.LayerNorm(n_hidden_1),
            nn.Dropout(p=0.1, inplace=False),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.empty(num_numerical_types, dim))
        init.xavier_uniform_(self.weights)
        init.constant_(self.biases, 0.)

    def forward(self, x):
        x = x.unsqueeze(-1)  
        return x * self.weights + self.biases

class SPT(nn.Module):

    def __init__(self, args, device):
        super(SPT, self).__init__()
        self.args = args 
        dim = args.dim
        num_special_tokens = args.num_special_tokens
        categories = args.categories
        out = args.out
        self.out = out
        self.num_cont = args.num_cont
        num_cont = args.num_cont

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        total_tokens = self.num_unique_categories + num_special_tokens

        if self.num_unique_categories > 0:
            categories_offset = torch.nn.functional.pad(torch.tensor(categories), (1, 0), value=num_special_tokens)
            self.register_buffer('categories_offset', categories_offset.cumsum(-1)[:-1])
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        if num_cont > 0:
            self.numerical_embedder = NumericalEmbedder(dim, num_cont)

        e_layers = args.e_layers
        self.layer = e_layers
        dropout = args.ff_dropout
        self.seq = self.num_cont + self.num_categories
        self.model = nn.ModuleList([SpecBlock(self.seq, self.seq // 2, self.seq // 2, top_k=5,
                                            d_model=dim, d_ff=dim, num_kernels=6, device=device) for _ in
                                    range(e_layers)])

        self.enc_embedding = DataEmbedding(args.dim, args.dim)
        self.layer_norm = nn.LayerNorm(dim)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.is_multi_task = hasattr(args, 'multi_task') and isinstance(args.multi_task, list) and len(args.multi_task) > 0
        
        feature_dim = dim * self.seq
        
        if self.is_multi_task:
            self.heads = nn.ModuleDict()
            self.loss_log_vars = nn.ParameterDict()
            for t_name in args.multi_task:
                if hasattr(args, t_name):
                    t_conf = getattr(args, t_name)
                    if t_conf['mode'] == 'Reg':
                        t_out = 1
                    else:
                        t_out = t_conf['num']
                else:
                    t_out = 1
                
                self.heads[t_name] = Prediction(feature_dim, 256, 128, t_out)
                self.loss_log_vars[t_name] = nn.Parameter(torch.zeros(1))
        else:
            self.projection = Prediction(feature_dim, 256, 128, out) 

    def model_name(self):
        return 'SPT'

    def get_shared_features(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)
        # [Batch, Seq, Dim] -> [Batch, Seq*Dim]
        output = output.reshape(output.shape[0], -1)
        return output

    def forward(self, x_categ, x_numer, label, step=None, return_attn=False, delta=False):
        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            x_categ_embed = self.categorical_embeds(x_categ)
            xs.append(x_categ_embed)
        
        if self.num_cont > 0:
            x_numer_embed = self.numerical_embedder(x_numer)
            xs.append(x_numer_embed)

        x_enc = torch.cat(xs, dim=1)
        
        enc_feature = self.get_shared_features(x_enc)

        
        if self.is_multi_task:
            total_loss = 0.0
            outs = {} 
            
            for t_name, head_module in self.heads.items():
                pred = head_module(enc_feature)
                outs[t_name] = pred

                if label is not None and isinstance(label, dict) and t_name in label:
                    t_label = label[t_name]
                    
                    log_var = self.loss_log_vars[t_name]
                    precision = torch.exp(-log_var)

                    if pred.shape[1] == 1:
                        # --- Regression Task --- Loss = 0.5 * exp(-s) * MSE + 0.5 * s
                        raw_loss = nn.MSELoss()(pred.squeeze(-1), t_label.float())
                        weighted_loss = 0.5 * precision * raw_loss + 0.5 * log_var
                        # weighted_loss = nn.MSELoss()(pred.squeeze(-1), t_label.float())
                    else:
                        # --- Classification Task --- Loss = exp(-s) * CE + 0.5 * s
                        raw_loss = nn.CrossEntropyLoss()(pred, t_label)
                        weighted_loss = precision * raw_loss + 0.5 * log_var
                        # weighted_loss = nn.CrossEntropyLoss()(pred, t_label)
                    
                    total_loss += weighted_loss
            
            return outs, total_loss

        else:
            dec_out = self.projection(enc_feature)

            if label is not None:
                if self.out == 1:
                    loss = nn.MSELoss()(dec_out.squeeze(-1), label.float())
                else:
                    loss = nn.CrossEntropyLoss()(dec_out, label)
            else:
                loss = None 
                
            return dec_out, loss