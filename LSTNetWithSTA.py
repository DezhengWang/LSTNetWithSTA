import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = args.feature
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip

        self.is_SSA = args.is_SSA
        self.is_TSA = args.is_TSA

        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = args.highway_window
        self.is_modified = args.is_modified
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p = args.dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            if self.is_SSA:
                self.highway = nn.Linear(153, 1)
            else:
                self.highway = nn.Linear(self.hw, 1)

        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh
        if (args.output_fun == 'relu'):
            self.output = F.relu

        # Attention
        if self.is_SSA:
            self.encoder_f = nn.LSTM(input_size=self.P, hidden_size=128, num_layers=1, batch_first=True)
            self.attention_f = ScaledDotProductAttention(d_model=128, d_k=128, d_v=128, h=2)
            self.alpha = nn.Parameter(torch.Tensor(1),requires_grad=True)

        if self.is_TSA:
            self.attention_t = ScaledDotProductAttention(d_model=self.m, d_k=self.m, d_v=self.m, h=6)
            self.belta = nn.Parameter(torch.Tensor(1),requires_grad=True)

            self.attention_r = ScaledDotProductAttention(d_model=100, d_k=100, d_v=100, h=1)
            self.gamma_r = nn.Parameter(torch.Tensor(1),requires_grad=True)

            self.attention_g = ScaledDotProductAttention(d_model=128, d_k=128, d_v=128, h=1)
            self.gamma_g = nn.Parameter(torch.Tensor(1),requires_grad=True)

            self.attention_l = ScaledDotProductAttention(d_model=45, d_k=45, d_v=45, h=1)
            self.gamma_f = nn.Parameter(torch.Tensor(1),requires_grad=True)
 
    def forward(self, x):
        batch_size = x.size(0)
        att2t, att2f = torch.tensor(0), torch.tensor(0)
        # Attention
        if self.is_TSA:
            att2t = self.attention_t(x, x, x)
            x = x + self.belta * att2t
        if self.is_SSA:
            x_f, _ = self.encoder_f(x.permute(0, 2, 1))
            att2f = self.attention_f(x_f, x_f, x_f)
        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # Attention
        if self.is_TSA:
            att2r = self.attention_r(c.permute(0,2,1), c.permute(0,2,1), c.permute(0,2,1))
            c = c.permute(0,2,1) + self.gamma_r * att2r
            c = c.permute(0,2,1)

        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)

        # Attention
        if self.is_TSA:
            att2g = self.attention_g(r, r, r)
            r = r + self.gamma_g * att2g

        r = self.dropout(torch.squeeze(r,0))

        #skip-rnn
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)

            # Attention
            if self.is_TSA:
                att2g = self.attention_l(s.unsqueeze(1), s.unsqueeze(1), s.unsqueeze(1))
                s = s.unsqueeze(1) + self.gamma_f * att2g
                s = s.squeeze(1)

            s = self.dropout(s)
            r = torch.cat((r,s),1)

        res = self.linear1(r)

        #highway
        if (self.hw > 0):
            x = x.permute(0, 2, 1).contiguous().view(-1, self.hw)
            # Attention
            if self.is_SSA:
                z = x_f + self.alpha * att2f
                z = z.contiguous().view(-1, 128)
                z = self.highway(torch.cat((x,z),dim=1))
            else:
                z = x
                z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res, att2t, att2f
    
class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v=0, h=1,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        if d_v!=0:
            self.fc_v = nn.Linear(d_model, h * d_v)
            self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)
        self.relu = F.relu

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values=None, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        if values is not None:
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)
        # att = att.squeeze(1)
        if values is not None:
            out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
            out = self.fc_o(out)  # (b_s, nq, d_model)
            return self.relu(out)
        return att
