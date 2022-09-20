import pickle

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from .resnet import *
from .graph import normalize_digraph
from .basic_block import *

def resnet18_AU(pretrained=True, **kwargs): # Constructs a ResNet-18 model
    model = ResNet_AU(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        resnet18_path = '/media/data1/wf/GFE2N/codes/codes4/materials/resnet18-raw.pth'
        pre_state_dict = torch.load(resnet18_path)#['state_dict']
        model.load_state_dict(pre_state_dict, False)
    return model

def resnet18_EMO(pretrained=True, **kwargs): # Constructs a ResNet-18 model
    model = ResNet_EMO(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        resnet18_path = '/media/data1/wf/GFE2N/codes/codes4/materials/resnet18-raw.pth'
        pre_state_dict = torch.load(resnet18_path)#['state_dict']
        model.load_state_dict(pre_state_dict, False)
    return model

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
    def __init__(self, args, pretrained=True, num_classes=7):
        super(EAC, self).__init__()
        resnet18 = resnet18_EMO()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(*list(resnet18.children())[:-2])  
        self.features2 = nn.Sequential(*list(resnet18.children())[-2:-1])  
        self.fc = nn.Linear(512, num_classes)  
        
        
    def forward(self, x):        
        x = self.features(x) # 1, 512, 7, 7
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
    def __init__(self, num_classes=12, neighbor_num=4, metric='dots'):
        super(GraphAU, self).__init__()
        self.backbone = resnet18_AU()
        self.in_channels = self.backbone.fc.weight.shape[1]
        self.out_channels = self.in_channels // 4
        self.backbone.fc = None

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_classes, neighbor_num, metric)

    def forward(self, x):
        # x: b d c
        x = self.backbone(x) # 1 * 49 * 512
        x = self.global_linear(x)
        cl = self.head(x)
        return cl
