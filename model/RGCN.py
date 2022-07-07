import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.sparse.linalg import eigs
from model.att import AttLayer

class AVWGCN(nn.Module):
    def __init__(self, cheb_polynomials, L_tilde, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.cheb_polynomials = cheb_polynomials
        self.L_tilde = L_tilde
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        
        # for existing graph convolution
        # self.init_gconv = nn.Conv1d(dim_in, dim_out, kernel_size=5, padding=0)
        self.init_gconv = nn.Linear(dim_in, dim_out)
        self.gconv = nn.Linear(dim_out * cheb_k, dim_out)
        self.dy_gate1 = AttLayer(dim_out)
        self.dy_gate2 = AttLayer(dim_out)

    def forward(self, x, node_embeddings, L_tilde_learned):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        b, n, _ = x.shape
        # 0) learned cheb_polynomials
        node_num = node_embeddings.shape[0]

        # L_tilde_learned = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        # L_tilde_learned = torch.matmul(L_tilde_learned, self.L_tilde) * L_tilde_learned

        support_set = [torch.eye(node_num).to(L_tilde_learned.device), L_tilde_learned]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * L_tilde_learned, support_set[-1]) - support_set[-2])

        # 1) convolution with learned graph convolution (implicit knowledge)
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv0 = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out

        # 2) convolution with existing graph (explicit knowledge)
        graph_supports = torch.stack(self.cheb_polynomials, dim=0)  # [k, n, m]
        x = self.init_gconv(x)
        x_g1 = torch.einsum("knm,bmc->bknc", graph_supports, x)
        x_g1 = x_g1.permute(0, 2, 1, 3).reshape(b, n, -1)  # B, N, cheb_k, dim_in
        x_gconv1 = self.gconv(x_g1)

        # 3) fusion of explit knowledge and implicit knowledge
        x_gconv = self.dy_gate1(F.leaky_relu(x_gconv0).transpose(1,2)) + self.dy_gate2(F.leaky_relu(x_gconv1).transpose(1,2))
        # x_gconv = F.leaky_relu(x_gconv0) + F.leaky_relu(x_gconv1)
        
        return x_gconv.transpose(1,2)
