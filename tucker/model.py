import numpy as np
import pandas as pd
import torch
from torch.nn.init import xavier_normal_


def load_num_lit(ent2idx, dataset, wrap=True):
    df = pd.read_csv(f'data/{dataset}/numerical_literals.txt', header=None, sep='\t')
    rel2idx = {v: k for k, v in enumerate(df[1].unique())}
    numerical_literals = np.zeros([len(ent2idx), len(rel2idx)], dtype=np.float32)
    for i, (s, p, lit) in enumerate(df.values):
        try:
            numerical_literals[ent2idx[s.lower()], rel2idx[p]] = lit
        except KeyError:
            continue
    max_lit, min_lit = np.max(numerical_literals, axis=0), np.min(numerical_literals, axis=0)
    numerical_literals = (numerical_literals - min_lit) / (max_lit - min_lit + 1e-8)
    if wrap:
        return torch.autograd.Variable(torch.from_numpy(numerical_literals))
    return numerical_literals


def load_num_lit_kbln(ent2idx, rel2idx, dataset):
    def load_data(file_path, ent2idx, rel2idx):
        df = pd.read_csv(file_path, sep='\t', header=None)
        M = df.shape[0]  # dataset size
        X = np.zeros([M, 3], dtype=int)
        for i, row in df.iterrows():
            X[i, 0] = ent2idx[row[0]]
            X[i, 1] = rel2idx[row[1]]
            X[i, 2] = ent2idx[row[2]]
        return X
    # Compute numerical literals
    numerical_literals = load_num_lit(ent2idx, dataset, wrap=False)
    # Compute X_train
    X_train = load_data(f"data/{dataset}/train.txt", ent2idx, rel2idx).astype(np.int32)
    h, t = X_train[:, 0], X_train[:, 2]
    n = numerical_literals[h, :] - numerical_literals[t, :]
    c = np.mean(n, axis=0).astype('float32')
    var = np.var(n, axis=0) + 1e-6

    return torch.autograd.Variable(torch.from_numpy(numerical_literals)), \
           torch.autograd.Variable(torch.FloatTensor(c)), \
           torch.autograd.Variable(torch.FloatTensor(var))


class Gate(torch.nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 gate_activation=torch.sigmoid):

        super(Gate, self).__init__()
        self.output_size = output_size

        self.gate_activation = gate_activation
        self.g = torch.nn.Linear(input_size, output_size)
        self.g1 = torch.nn.Linear(output_size, output_size, bias=False)
        self.g2 = torch.nn.Linear(input_size-output_size, output_size, bias=False)
        self.gate_bias = torch.nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], 1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, device=kwargs["device"], requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def to_cuda(self):
        pass

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred


class TuckER_Literal(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER_Literal, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

        # Literal
        self.numerical_literals = load_num_lit(kwargs["ent2idx"], kwargs["dataset"])
        self.n_num_lit = self.numerical_literals.size(1)
        self.emb_num_lit = Gate(d1 + self.n_num_lit, d1)

    def to_cuda(self):
        self.numerical_literals = self.numerical_literals.cuda()
        self.emb_num_lit = self.emb_num_lit.cuda()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        # Begin Literals
        e1_num_lit = self.numerical_literals[e1_idx.view(-1)]
        e1 = self.emb_num_lit(e1, e1_num_lit)
        e2_multi_emb = self.emb_num_lit(self.E.weight, self.numerical_literals)

        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, e2_multi_emb.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred


class TuckER_KBLN(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER_KBLN, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

        # Literal
        self.num_entities = len(d.entities)
        self.numerical_literals, self.c, self.var = load_num_lit_kbln(kwargs["ent2idx"],
                                                                      kwargs["rel2idx"],
                                                                      kwargs["dataset"])
        self.n_num_lit = self.numerical_literals.size(1)
        self.nf_weights = torch.nn.Embedding(len(d.relations), self.n_num_lit)

    def to_cuda(self):
        self.numerical_literals = self.numerical_literals.cuda()
        self.c = self.c.cuda()
        self.var = self.var.cuda()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def rbf(self, n):
        """
        Apply RBF kernel parameterized by (fixed) c and var, pointwise.
        n: (batch_size, num_ents, n_lit)
        """
        return torch.exp(-(n - self.c)**2 / self.var)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        score_l = torch.mm(x, self.E.weight.transpose(1, 0))

        # Begin numerical literals
        n_h = self.numerical_literals[e1_idx.view(-1)]
        n_t = self.numerical_literals

        # Features (batch_size x num_ents x n_lit)
        n = n_h.unsqueeze(1).repeat(1, self.num_entities, 1) - n_t
        phi = self.rbf(n)
        # Weights (batch_size, 1, n_lits)
        w_nf = self.nf_weights(r_idx.view(-1, 1))

        score_n = torch.bmm(phi, w_nf.transpose(1, 2)).squeeze()
        # End numerical literals

        pred = torch.sigmoid(score_l + score_n)
        return pred


class DistMult(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(DistMult, self).__init__()

        assert(d1 == d2)

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def to_cuda(self):
        pass

    def forward(self, e1_idx, r_idx):
        e1 = self.input_dropout(self.E(e1_idx).squeeze())
        r = self.input_dropout(self.R(r_idx).squeeze())

        return torch.sigmoid(torch.mm(e1*r, self.E.weight.transpose(1, 0)))


class DistMult_Literal(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(DistMult_Literal, self).__init__()

        assert(d1 == d2)

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.loss = torch.nn.BCELoss()

        # Literal
        self.numerical_literals = load_num_lit(kwargs["ent2idx"], kwargs["dataset"])
        self.n_num_lit = self.numerical_literals.size(1)
        self.emb_num_lit = Gate(d1 + self.n_num_lit, d1)

    def to_cuda(self):
        self.numerical_literals = self.numerical_literals.cuda()
        self.emb_num_lit = self.emb_num_lit.cuda()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx).squeeze()
        r = self.R(r_idx).squeeze()

        # Begin literals
        e1_lit = self.numerical_literals[e1_idx.view(-1)]
        e1 = self.emb_num_lit(e1, e1_lit)
        e2 = self.emb_num_lit(self.E.weight, self.numerical_literals)
        # End literals

        e1 = self.input_dropout(e1)
        r = self.input_dropout(r)
        return torch.sigmoid(torch.mm(e1*r, e2.transpose(1, 0)))


class DistMult_KBLN(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(DistMult_KBLN, self).__init__()

        assert(d1 == d2)

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.loss = torch.nn.BCELoss()

        # Literal
        self.num_entities = len(d.entities)
        self.numerical_literals, self.c, self.var = load_num_lit_kbln(kwargs["ent2idx"],
                                                                      kwargs["rel2idx"],
                                                                      kwargs["dataset"])
        self.n_num_lit = self.numerical_literals.size(1)
        self.nf_weights = torch.nn.Embedding(len(d.relations), self.n_num_lit)

    def to_cuda(self):
        self.numerical_literals = self.numerical_literals.cuda()
        self.c = self.c.cuda()
        self.var = self.var.cuda()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def rbf(self, n):
        return torch.exp(-(n - self.c)**2 / self.var)

    def forward(self, e1_idx, r_idx):
        e1 = self.input_dropout(self.E(e1_idx).squeeze())
        r = self.input_dropout(self.R(r_idx).squeeze())
        score_l = torch.mm(e1*r, self.E.weight.transpose(1, 0))

        # Begin literals
        n_h = self.numerical_literals[e1_idx.view(-1)]
        n_t = self.numerical_literals

        n = n_h.unsqueeze(1).repeat(1, self.num_entities, 1) - n_t
        phi = self.rbf(n)
        w_nf = self.nf_weights(r_idx.view(-1, 1))

        score_n = torch.bmm(phi, w_nf.transpose(1, 2)).squeeze()
        # End literals

        return torch.sigmoid(score_l + score_n)


class ComplEx(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(ComplEx, self).__init__()

        assert(d1 == d2)

        self.dim = d1
        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def to_cuda(self):
        pass

    def forward(self, e1_idx, r_idx):
        e1_real = self.input_dropout(self.E(e1_idx)[:, :self.dim//2].squeeze())
        e1_img = self.input_dropout(self.E(e1_idx)[:, self.dim//2:].squeeze())
        r_real = self.input_dropout(self.R(r_idx)[:, :self.dim//2].squeeze())
        r_img = self.input_dropout(self.R(r_idx)[:, self.dim//2:].squeeze())

        realrealreal = torch.mm(e1_real*r_real, self.E.weight[:, :self.dim//2].transpose(1, 0))
        realimgimg = torch.mm(e1_real*r_img, self.E.weight[:, self.dim//2:].transpose(1, 0))
        imgrealimg = torch.mm(e1_img*r_real, self.E.weight[:, self.dim//2:].transpose(1, 0))
        imgimgreal = torch.mm(e1_img*r_img, self.E.weight[:, :self.dim//2].transpose(1, 0))

        return torch.sigmoid(realrealreal + realimgimg + imgrealimg - imgimgreal)


class ComplEx_Literal(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(ComplEx_Literal, self).__init__()

        assert(d1 == d2)

        self.dim = d1
        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.loss = torch.nn.BCELoss()

        # Literal
        self.numerical_literals = load_num_lit(kwargs["ent2idx"], kwargs["dataset"])
        self.n_num_lit = self.numerical_literals.size(1)
        self.emb_num_lit = Gate(d1 + self.n_num_lit, d1)

    def to_cuda(self):
        self.numerical_literals = self.numerical_literals.cuda()
        self.emb_num_lit = self.emb_num_lit.cuda()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1, r = self.E(e1_idx), self.R(r_idx)

        # Begin literals
        e1_lit = self.numerical_literals[e1_idx.view(-1)]
        e1 = self.emb_num_lit(e1, e1_lit)
        e2 = self.emb_num_lit(self.E.weight, self.numerical_literals)
        # End literals

        e1_real = self.input_dropout(e1[:, :self.dim//2].squeeze())
        e1_img = self.input_dropout(e1[:, self.dim//2:].squeeze())
        r_real = self.input_dropout(r[:, :self.dim//2].squeeze())
        r_img = self.input_dropout(r[:, self.dim//2:].squeeze())

        realrealreal = torch.mm(e1_real*r_real, e2[:, :self.dim//2].transpose(1, 0))
        realimgimg = torch.mm(e1_real*r_img, e2[:, self.dim//2:].transpose(1, 0))
        imgrealimg = torch.mm(e1_img*r_real, e2[:, self.dim//2:].transpose(1, 0))
        imgimgreal = torch.mm(e1_img*r_img, e2[:, :self.dim//2].transpose(1, 0))

        return torch.sigmoid(realrealreal + realimgimg + imgrealimg - imgimgreal)


class ComplEx_KBLN(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(ComplEx_KBLN, self).__init__()

        assert(d1 == d2)

        self.dim = d1
        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.loss = torch.nn.BCELoss()

        # Literal
        self.num_entities = len(d.entities)
        self.numerical_literals, self.c, self.var = load_num_lit_kbln(kwargs["ent2idx"],
                                                                      kwargs["rel2idx"],
                                                                      kwargs["dataset"])
        self.n_num_lit = self.numerical_literals.size(1)
        self.nf_weights = torch.nn.Embedding(len(d.relations), self.n_num_lit)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def to_cuda(self):
        self.numerical_literals = self.numerical_literals.cuda()
        self.c = self.c.cuda()
        self.var = self.var.cuda()

    def rbf(self, n):
        return torch.exp(-(n - self.c)**2 / self.var)

    def forward(self, e1_idx, r_idx):
        e1_real = self.input_dropout(self.E(e1_idx)[:, :self.dim//2].squeeze())
        e1_img = self.input_dropout(self.E(e1_idx)[:, self.dim//2:].squeeze())
        r_real = self.input_dropout(self.R(r_idx)[:, :self.dim//2].squeeze())
        r_img = self.input_dropout(self.R(r_idx)[:, self.dim//2:].squeeze())

        realrealreal = torch.mm(e1_real*r_real, self.E.weight[:, :self.dim//2].transpose(1, 0))
        realimgimg = torch.mm(e1_real*r_img, self.E.weight[:, self.dim//2:].transpose(1, 0))
        imgrealimg = torch.mm(e1_img*r_real, self.E.weight[:, self.dim//2:].transpose(1, 0))
        imgimgreal = torch.mm(e1_img*r_img, self.E.weight[:, :self.dim//2].transpose(1, 0))

        score_l = realrealreal + realimgimg + imgrealimg - imgimgreal

        # Begin literals
        n_h = self.numerical_literals[e1_idx.view(-1)]
        n_t = self.numerical_literals

        n = n_h.unsqueeze(1).repeat(1, self.num_entities, 1) - n_t
        phi = self.rbf(n)
        w_nf = self.nf_weights(r_idx.view(-1, 1))

        score_n = torch.bmm(phi, w_nf.transpose(1, 2)).squeeze()
        # End literals

        return torch.sigmoid(score_l + score_n)


class ConvE(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(ConvE, self).__init__()

        assert(d1 == d2)

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.feature_map_dropout = torch.nn.Dropout2d(kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=kwargs["use_bias"])
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(len(d.entities))))
        self.fc = torch.nn.Linear(kwargs["hidden_size"], d1)

        self.emb_dim1 = kwargs["embedding_shape1"]
        self.emb_dim2 = d1 // self.emb_dim1

    def to_cuda(self):
        pass

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):

        e1 = self.E(e1_idx).view(-1, 1, self.emb_dim1, self.emb_dim2)
        r = self.R(r_idx).view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = self.bn0(torch.cat([e1, r], 2))

        x = self.input_dropout(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        return torch.sigmoid(x)


class ConvE_Literal(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(ConvE_Literal, self).__init__()

        assert(d1 == d2)

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.feature_map_dropout = torch.nn.Dropout2d(kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=kwargs["use_bias"])
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(len(d.entities))))
        self.fc = torch.nn.Linear(kwargs["hidden_size"], d1)

        self.emb_dim1 = kwargs["embedding_shape1"]
        self.emb_dim2 = d1 // self.emb_dim1

        # Literal
        self.numerical_literals = load_num_lit(kwargs["ent2idx"], kwargs["dataset"])
        self.n_num_lit = self.numerical_literals.size(1)
        self.emb_num_lit = Gate(d1 + self.n_num_lit, d1)

    def to_cuda(self):
        self.numerical_literals = self.numerical_literals.cuda()
        self.emb_num_lit = self.emb_num_lit.cuda()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):

        e1 = self.E(e1_idx)

        # Begin literals
        e1_lit = self.numerical_literals[e1_idx.view(-1)]
        e1 = self.emb_num_lit(e1, e1_lit).view(-1, 1, self.emb_dim1, self.emb_dim2)
        e2 = self.emb_num_lit(self.E.weight, self.numerical_literals)
        # End literals

        r = self.R(r_idx).view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = self.bn0(torch.cat([e1, r], 2))

        x = self.input_dropout(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = torch.mm(x, e2.transpose(1, 0))
        x += self.b.expand_as(x)
        return torch.sigmoid(x)


class ConvE_KBLN(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(ConvE_KBLN, self).__init__()

        assert(d1 == d2)

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.feature_map_dropout = torch.nn.Dropout2d(kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=kwargs["use_bias"])
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(len(d.entities))))
        self.fc = torch.nn.Linear(kwargs["hidden_size"], d1)

        self.emb_dim1 = kwargs["embedding_shape1"]
        self.emb_dim2 = d1 // self.emb_dim1

        # Literal
        self.num_entities = len(d.entities)
        self.numerical_literals, self.c, self.var = load_num_lit_kbln(kwargs["ent2idx"],
                                                                      kwargs["rel2idx"],
                                                                      kwargs["dataset"])
        self.n_num_lit = self.numerical_literals.size(1)
        self.nf_weights = torch.nn.Embedding(len(d.relations), self.n_num_lit)

    def to_cuda(self):
        self.numerical_literals = self.numerical_literals.cuda()
        self.c = self.c.cuda()
        self.var = self.var.cuda()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def rbf(self, n):
        return torch.exp(-(n - self.c)**2 / self.var)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx).view(-1, 1, self.emb_dim1, self.emb_dim2)
        r = self.R(r_idx).view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = self.bn0(torch.cat([e1, r], 2))

        x = self.input_dropout(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        x += self.b.expand_as(x)

        # Begin literals
        n_h = self.numerical_literals[e1_idx.view(-1)]
        n_t = self.numerical_literals

        n = n_h.unsqueeze(1).repeat(1, self.num_entities, 1) - n_t
        phi = self.rbf(n)
        w_nf = self.nf_weights(r_idx.view(-1, 1))

        score_n = torch.bmm(phi, w_nf.transpose(1, 2)).squeeze()
        # End literals

        return torch.sigmoid(x + score_n)
