import torch
import torch.nn.functional as F
def weighted_cross_entropy_loss(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """

    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    # edges= torch.cat([edge,edge,edge,edge,edge,edge,edge], dim=0)
    # print(preds.shape, edges.shape)
    mask = (edges > 0.5).float()
    b, c, h, w = mask.shape

    # Shape: [b,].
    num_pos = torch.sum(mask, dim=[1,2,3], keepdim=True).float()
    # print("pos", num_pos.shape)

    num_neg = c*h * w - num_pos                     # Shape: [b,].
    # print("neg", num_neg.shape)
    weight = torch.zeros_like(mask)
    #weight[edges > 0.5]  = num_neg / (num_pos + num_neg)
    #weight[edges <= 0.5] = num_pos / (num_pos + num_neg)
    weight.masked_scatter_(edges > 0.5,
                           torch.ones_like(edges) * num_neg / (num_pos + num_neg))
    weight.masked_scatter_(edges <= 0.5,
                           torch.ones_like(edges) * num_pos / (num_pos + num_neg))

    # Calculate loss
    # preds=torch.sigmoid(preds)
    losses = F.binary_cross_entropy_with_logits(preds,
                                                edges,
                                                weight=weight,
                                                reduction='none')
    losses=losses.sum(dim=[1,2,3],keepdim=True) / (c * h * w)

    return torch.mean(losses)
