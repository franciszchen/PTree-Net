import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data



class PatchRelevance(torch.nn.Module):
    """ inner product of semantic features to describe context for each node
    """
    def __init__(self, in_dim, out_dim):
        super(PatchRelevance, self).__init__()
        self.lin_q = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_k = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_v = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        #
        x_q = self.lin_q(x) # N*C
        x_k = self.lin_k(x) # N*C
        x_k = x_k.permute(1, 0) # C*N
        vertex_affinity = torch.mm(x_q, x_k) # N*N, equation 3
        vertex_affinity = F.softmax(vertex_affinity, dim=-1)
        x_v = self.lin_v(x) # N*C
        return torch.mm(vertex_affinity, x_v) # equation 4
        
class Aggregator(torch.nn.Module):
    """
    RGCN
    """
    def __init__(self, class_num=4, in_dim=512, inter_dim=128,  out_dim=64):
        super(Aggregator, self).__init__()
        self.fc_tw = nn.Linear(out_dim, class_num, bias=True)
        self.fc_mid = nn.Linear(out_dim, class_num, bias=True)
        self.fc_bottom = nn.Linear(out_dim, class_num, bias=True)

        self.gconv1 = GCNConv(in_dim, inter_dim)
        self.gconv2 = GCNConv(inter_dim, out_dim)
        
        self.patch_relevance1 = PatchRelevance(in_dim=in_dim, out_dim=inter_dim)
        self.patch_relevance2 = PatchRelevance(in_dim=inter_dim, out_dim=out_dim)
        
        self.squeeze1 = nn.Linear(int(2*inter_dim), inter_dim, bias=True)
        self.squeeze2 = nn.Linear(int(2*out_dim), out_dim, bias=True)

        self.fc1 = nn.Linear(int(3*out_dim), out_dim, bias=True)
        self.fc2 = nn.Linear(out_dim, class_num, bias=True)

        self.sparse_attention = nn.Conv1d(out_dim, 4, kernel_size=4, stride=4, bias=True)

    def forward(self, x, edge_index, arch_list):
        relevance1 = self.patch_relevance1(x)
        x_gconv1 = self.gconv1(x, edge_index=edge_index)
        x_cat1 = F.relu(
                torch.cat([x_gconv1, relevance1], dim=1)
            ) # concat gconv1 and relevance1 
        x_sq1 = F.relu(
                self.squeeze1(x_cat1)
            ) # -> inter_dim

        relevance2 = self.patch_relevance2(x_sq1) 
        x_gconv2 = self.gconv2(x_sq1, edge_index) # equation 5
        x_cat2 = F.relu(
                torch.cat([x_gconv2, relevance2], dim=1)
            ) # concat gconv2 and relevance2 
        x_sq2 = F.relu(
                self.squeeze2(x_cat2)
            ) # -> out_dim
        
        """ weighted pooling """
        tw_feature = x_sq2[0, :].view(1, -1) # -> out_dim
        mid_features = x_sq2[1:1+arch_list[0], :].view(arch_list[0], -1)
        mid_average_feature = torch.mean(mid_features, dim=0, keepdim=True)
        bottom_features = x_sq2[1+arch_list[0]:1+arch_list[0]+arch_list[1], :]

        bottom_features_reshape = torch.transpose(torch.unsqueeze(bottom_features, dim=0), dim0=1, dim1=2)
        bottom_weights = self.sparse_attention(bottom_features_reshape)
        bottom_weights = bottom_weights.view(4, -1)

        bottom_weights_vector = torch.zeros(arch_list[1]).cuda()
        for patch_idx in range(4):
            index = torch.tensor(list(range(patch_idx, arch_list[1], 4))).cuda()
            bottom_weights_vector.index_copy_(0, index, bottom_weights[patch_idx])

        bottom_weights_vector = (
            bottom_weights_vector/torch.max(bottom_weights_vector)
            ).view(-1, 1)

        bottom_weights_vector = (bottom_weights_vector.view(arch_list[0], 4)/torch.sum(torch.abs(bottom_weights_vector.view(arch_list[0], 4)), dim=-1, keepdim=True)).view(arch_list[1], 1)
        tree_bottom_features = torch.transpose(
            torch.sum(
                torch.transpose(
                    bottom_weights_vector.view(-1, 1)*bottom_features, 0, 1
                    ).view(-1, arch_list[0], 4), 
                dim=-1, keepdim=False
            ), 
            0, 1
        )

        bottom_weights_vector = bottom_weights_vector.view(-1, 1)

        bottom_weighted_feature = torch.sum(
            bottom_weights_vector.view(-1, 1)*bottom_features, dim=0, keepdim=False
            ).view(1, -1) / arch_list[0]
        
        """
        3level out feature vectors: tw, mid, bottom -> reduce the node-dim into 1
        then cat them togther for final gcn predict
        sparse weighted aggregation 
        """
        out_tw = self.fc_tw(tw_feature).view(1, -1)
        out_mid = self.fc_mid(mid_average_feature).view(1, -1)
        out_bottom = self.fc_bottom(
            bottom_weighted_feature
            ).view(1, -1)
        gcn_feature = self.fc1(
            torch.cat(
                [tw_feature, mid_average_feature, bottom_weighted_feature], dim=-1) 
            )
        out_gcn = self.fc2(
                F.relu(
                    gcn_feature
                )
            )
        return out_gcn, out_tw, out_mid, out_bottom, \
            gcn_feature, tw_feature, mid_average_feature, bottom_weighted_feature, bottom_weights_vector, \
                mid_features, tree_bottom_features

if __name__ == '__main__':
    
    context = PatchRelevance(in_dim=512, out_dim=128)
    x = torch.rand(size=(23, 512))

    out = context(x)
    print('out shape:\t', out.shape)