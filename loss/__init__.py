from torch.nn.functional import binary_cross_entropy_with_logits as BCEWithLogitsLoss
from torch.nn.functional import cross_entropy as CrossEntropyLoss
from .extra_losses import weighted_cross_entropy_loss