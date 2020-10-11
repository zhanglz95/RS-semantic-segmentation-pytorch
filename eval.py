import torch 
from tqdm import tqdm

def eval_net(model, loader, num_classes, device):
    model.eval()
    mask_type = torch.float32 if num_classes == 1 else torch.long
    n_val = len(loader)
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch') as pbar:
        for imgs, true_masks in loader:
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                pred_masks = model(imgs)
            
            if num_classes > 1:
                raise NotImplementedError
            else:
                pred = torch.sigmoid(pred_masks)
                pred = (pred > 0.5).float()
                tot += diceCoeff(true_masks, pred)
            pbar.update()
    model.train()
    return tot / n_val

def diceCoeff(gt, mask):
    if gt.is_cuda:
        sum = torch.FloatTensor(1).cuda().zero_()
    else:
        sum = torch.FloatTensor(1).zero_()
    
    eps = 1e-4
    inter = torch.dot(gt.view(-1), mask.view(-1))
    union = torch.sum(gt) + torch.sum(mask) + eps

    return (inter.float() * 2 + eps) / union.float()