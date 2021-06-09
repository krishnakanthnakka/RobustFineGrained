import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Variable


class Attention_pooling(nn.Module):

    def __init__(self, feat_dim, num_classes):

        super(Attention_pooling, self).__init__()
        self.class_agnostic_weights = nn.Conv2d(feat_dim, 1, kernel_size=1)
        self.class_specific_weights = nn.Conv2d(
            feat_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout(p=0.8)
        self.returnpool_dim = num_classes
        self.relu = nn.ReLU()
        self.classmix_att = None

    def forward(self, x, return_endpoints=False):
        conv_map_relu = F.relu(x)
        ca_logits = self.class_agnostic_weights(conv_map_relu)
        conv_map_relu = self.dropout(conv_map_relu)
        cs_logits = self.class_specific_weights(conv_map_relu)
        att_logits = torch.mean(cs_logits * ca_logits, dim=[2, 3])
        end_points = {}
        end_points['ca_att'] = ca_logits
        end_points['cs_att'] = cs_logits

        logits_comb = self.relu(ca_logits * cs_logits)
        end_points['cas_att'] = logits_comb

        logits_comb = torch.max(logits_comb, dim=1, keepdim=True)[0]
        max_clsmix = torch.max(logits_comb, dim=2, keepdim=True)[0]
        max_clsmix = torch.max(max_clsmix, dim=3, keepdim=True)[0]
        logits_comb = logits_comb / max_clsmix
        self.classmix_att = logits_comb
        end_points['classmix_att'] = logits_comb
        end_points['att_logits'] = att_logits
        return att_logits, end_points if return_endpoints else att_logits

    def get_classmixattention(self):
        return self.classmix_att


def get_pooling_layer(feat_dim, num_classes, return_attention_branch=False):

    return Attention_pooling(feat_dim, num_classes)
