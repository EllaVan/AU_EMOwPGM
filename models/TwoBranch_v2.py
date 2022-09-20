import pickle

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from .resnet import *
from .graph import normalize_digraph
from .basic_block import *

from simclr import SimCLR
from simclr.modules.identity import Identity

class Head(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(Head, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.gnn = GNN(self.in_channels, self.num_classes,neighbor_num=neighbor_num,metric=metric)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        # FGG
        f_v = self.gnn(f_v)
        b, n, c = f_v.shape
        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1)
        cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        return cl

class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(GNN, self).__init__()
        # in_channels: dim of node feature
        # num_classes: num of nodes
        # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
        # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
        # X' = ReLU(X + BN(V(X) + A x U(X)) )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(self.in_channels,self.in_channels)
        self.V = nn.Linear(self.in_channels,self.in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)

        # init
        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape

        # build dynamical graph
        if self.metric == 'dots':
            si = x.detach()
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'cosine':
            si = x.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'l1':
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si)
            si = si.sum(dim=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()

        else:
            raise Exception("Error: wrong metric: ", self.metric)

        # GNN process
        A = normalize_digraph(adj)
        aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
        x = self.relu(x + self.bnv(aggregate + self.U(x)))
        return x

class EAC(nn.Module):
    def __init__(self, num_classes=7):
        super(EAC, self).__init__()
        self.num_classes = num_classes
        self.features2 = self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)  
        
    def forward(self, x):  
        feature = self.features2(x) # 1, 512, 1, 1
        feature = feature.view(feature.size(0), -1)
        output = self.fc(feature)
        params = list(self.parameters())
        fc_weights = params[-2].data
        fc_weights = fc_weights.view(1, self.num_classes, 512, 1, 1)
        fc_weights = Variable(fc_weights, requires_grad = False)
        # attention
        feat = x.unsqueeze(1) # N * 1 * C * H * W
        hm = feat * fc_weights
        hm = hm.sum(2) # N * self.num_labels * H * W
        return output, hm

class GraphAU(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=12, neighbor_num=4, metric='dots'):
        super(GraphAU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.neighbor_num = neighbor_num
        self.metric = metric
        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_classes, neighbor_num, metric)

    def forward(self, x):
        # x: b d c
        b,c,h,w = x.shape
        x = x.view(b,c,-1).permute(0,2,1)
        x = self.global_linear(x)
        cl = self.head(x)
        return cl

class SSL_encoder(nn.Module):
    def __init__(self, n_features, projection_dim):
        super(SSL_encoder, self).__init__()
        self.n_features = n_features
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # Replace the fc layer with an Identity function
        # self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        x_i = self.avgpool(x_i)
        x_j = self.avgpool(x_j)
        h_i = x_i.view(x_i.size(0), -1)
        h_j = x_j.view(x_j.size(0), -1)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j

class TwoBranch_SSL_Pred(nn.Module):
    def __init__(self, pretrained=True, num_AUs=12, num_EMOs=7, neighbor_num=4, metric='dots', ispred_AU=True, ispred_EMO=True, **kwargs):
        super(TwoBranch_SSL_Pred, self).__init__()
        self.ispred_AU = ispred_AU
        self.ispred_EMO = ispred_EMO
        self.backbone = ResNet_backbone(BasicBlock, [2, 2, 2, 2], **kwargs)
        self.projection_dim = 64
        n_features = self.backbone.fc.in_features
        self.encoder = SSL_encoder(n_features, self.projection_dim)
        if pretrained:
            resnet18_path = '/media/data1/wf/GFE2N/codes/codes4/materials/resnet18-raw.pth'
            pre_state_dict = torch.load(resnet18_path)#['state_dict']
            self.backbone.load_state_dict(pre_state_dict, False)
        if self.ispred_EMO is True:
            self.pred_EMO = EAC(num_EMOs)
        if self.ispred_AU is True:
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.pred_AU = GraphAU(self.in_channels, self.out_channels, num_AUs, neighbor_num, metric)
        
    def forward(self, x, x_SSL=None):        
        x = self.backbone(x) # 1, 512, 7, 7
        out_SSL = None
        if x_SSL is not None:
            x_SSL = self.backbone(x_SSL) # 1, 512, 7, 7
            out_SSL = self.encoder(x, x_SSL)
        out_pred = []
        if self.ispred_AU is True:
            cl = self.pred_AU(x)
            out_pred.append(cl)
        if self.ispred_EMO is True:
            output, hm = self.pred_EMO(x)
            out_pred.append(output)
            out_pred.append(hm)
        return out_SSL, tuple(out_pred)

class TwoBranch_Pred(nn.Module):
    def __init__(self, pretrained=True, num_AUs=12, num_EMOs=7, neighbor_num=4, metric='dots', ispred_AU=True, ispred_EMO=True, **kwargs):
        super(TwoBranch_Pred, self).__init__()
        self.ispred_AU = ispred_AU
        self.ispred_EMO = ispred_EMO
        self.backbone = ResNet_backbone(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            # resnet18_path = '/media/data1/wf/AU_EMOwPGM/codes/results/BP4D/SSL/subject_independent/bs_64_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/epoch20_model_fold0.pth'
            # pre_state_dict = torch.load(resnet18_path, map_location='cpu')['state_dict']
            # state_dict = {}
            # for k, v in pre_state_dict.items():
            #     if k.split('.')[0] == 'encoder':
            #         k = k.replace('encoder.', '')
            #         state_dict[k] = v
            resnet18_path = '/media/data1/wf/GFE2N/codes/codes4/materials/resnet18-raw.pth'
            state_dict = torch.load(resnet18_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, False)
        if self.ispred_EMO is True:
            self.pred_EMO = EAC(num_EMOs)
        if self.ispred_AU is True:
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.pred_AU = GraphAU(self.in_channels, self.out_channels, num_AUs, neighbor_num, metric)
        
    def forward(self, x):        
        x = self.backbone(x) # 1, 512, 7, 7
        out_pred = []
        if self.ispred_AU is True:
            cl = self.pred_AU(x)
            out_pred.append(cl)
        if self.ispred_EMO is True:
            output, hm = self.pred_EMO(x)
            out_pred.append(output)
            out_pred.append(hm)
        return tuple(out_pred)

class GraphAU_SSL(nn.Module):
    def __init__(self, num_classes=12, neighbor_num=4, metric='dots', **kwargs):
        super(GraphAU_SSL, self).__init__()
        self.backbone = ResNet_backbone(BasicBlock, [2, 2, 2, 2], **kwargs)
        # if pretrained:
        resnet18_path = '/media/data1/wf/AU_EMOwPGM/codes/results/BP4D/SSL/subject_independent/bs_64_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/epoch20_model_fold0.pth'
        pre_state_dict = torch.load(resnet18_path, map_location='cpu')['state_dict']
        state_dict = {}
        for k, v in pre_state_dict.items():
            if k.split('.')[0] == 'encoder':
                k = k.replace('encoder.', '')
                state_dict[k] = v
        # resnet18_path = '/media/data1/wf/GFE2N/codes/codes4/materials/resnet18-raw.pth'
        # state_dict = torch.load(resnet18_path, map_location='cpu')
        self.backbone.load_state_dict(state_dict, False)
        self.in_channels = self.backbone.fc.weight.shape[1]
        self.out_channels = self.in_channels // 4
        self.pred_AU = GraphAU(self.in_channels, self.out_channels, num_classes, neighbor_num, metric)
        
    def forward(self, x):        
        x = self.backbone(x) # 1, 512, 7, 7
        cl = self.pred_AU(x)
        return cl

class EAC_SSL(nn.Module):
    def __init__(self,args, pretrained=True, num_classes=7):
        super(EAC_SSL, self).__init__()
        self.backbone = ResNet_backbone(BasicBlock, [2, 2, 2, 2])
        # if pretrained:
        resnet18_path = '/media/data1/wf/AU_EMOwPGM/codes/results/BP4D/SSL/subject_independent/bs_64_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/epoch20_model_fold0.pth'
        pre_state_dict = torch.load(resnet18_path, map_location='cpu')['state_dict']
        state_dict = {}
        for k, v in pre_state_dict.items():
            if k.split('.')[0] == 'encoder':
                k = k.replace('encoder.', '')
                state_dict[k] = v
        # resnet18_path = '/media/data1/wf/GFE2N/codes/codes4/materials/resnet18-raw.pth'
        # state_dict = torch.load(resnet18_path, map_location='cpu')
        self.backbone.load_state_dict(state_dict, False)
        self.pred_EMO = EAC(num_classes)
        
    def forward(self, x):        
        x = self.backbone(x) # 1, 512, 7, 7
        out_pred = []
        output, hm = self.pred_EMO(x)
        return output, hm


if __name__=='__main__':
    encoder = ResNet_backbone(BasicBlock, [2, 2, 2, 2])
    projection_dim = 64
    n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer
    model = SimCLR(encoder, projection_dim, n_features)


