import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn.inits import reset
from typing import Union, Tuple
from torch_geometric.nn import NNConv,Set2Set
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class SAGEConv(MessagePassing):
   
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            #print('节点维度',x[0].size(-1))
            #print('边维度',edge_attr.size(-1))
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        #print('hhhhhhhhhhh')
        #print(edge_index.dtype)
        #print(edge_attr.dtype)
        #print(out.dtype)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.relu(x_j + edge_attr)


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GraphSAGE,self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        #edge_attr = torch.mm(edge_attr, emb_ea)
        
        #print(x.dtype,adj_t.dtype,edge_attr.dtype)
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_attr)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            #print(x[0].size(-1))
        x = self.convs[-1](x, adj_t, edge_attr)
        return x

class DDI_MLP(nn.Module):
    def __init__(self,input_feature_dim,num_types,dim):
        super(DDI_MLP,self).__init__()
        self.classifer=nn.Sequential(
            nn.Linear(2*input_feature_dim,dim),
            
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim,num_types)
        )

    def forward(self,h,edge_index,edge_attr):
        h1=h.index_select(0,edge_index[0])#按照edge_index第一行中各个节点的序号，从h中找到这个头节点的特征向量，以DeepDDI数据集10%边的训练集为例，h维度为1861*50，h1的维度则变为17726*50
        h2=h.index_select(0,edge_index[1])#按照edge_index第二行中各个节点的序号，从h中找到这个尾节点的特征向量
        h3=torch.cat([h1,h2],-1)#进行拼接操作,h1和h2两个矩阵进行横向拼接，h3的维度变为17726*100
        #h3=h1*h2
        output=self.classifer(h3)#h3再经过MLP的训练即可以将维度映射到17726*113
        return F.sigmoid(output) #再经过sigmoid函数将数值转化为0-1之间

#Decoder部分实际上就是一层上面定义的MLP将encoder编码得到的h，解码得到边预测的类型（具体如何实现见MLP部分注释）
class DDIDecoder(nn.Module):
    def __init__(self,num_types,dim):
        super(DDIDecoder,self).__init__()
        self.model=DDI_MLP(dim,num_types,dim)

    def forward(self, h, edge_index, edge_attr=None):
        return self.model(h, edge_index, edge_attr)

class DDINet(nn.Module):
    def __init__(self, encoder, decoder, decoder2=None):
        super(DDINet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder  # used for inference
        self.decoder2 = decoder2  # used for test phase in an energy model

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)
        reset(self.decoder2)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilties."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, h, edge_index, edge_attr):
        out = self.decoder(h, edge_index, edge_attr)
        #再使用交叉熵损失函数计算模型的损失
        loss = F.binary_cross_entropy(out, edge_attr)
        return loss
