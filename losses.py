# losses.py
# used for training the model, which is a combination of binary cross-entropy loss and focal loss.
# The binary cross-entropy loss is used to measure the difference between the predicted probabilities and the true labels, 
# while the focal loss is used to address class imbalance by down-weighting easy examples and focusing on hard examples. 
# The combination of these two losses helps to improve the model's performance on imbalanced datasets, such as fraud detection.
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    # used for rebalancing the loss function to focus more on hard-to-classify examples, which is particularly useful in imbalanced datasets.
    # (1 in 22) of the postings are scams, so the model needs to be able to identify these rare cases effectively.
    def __init__(self, gamma=2.0, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma # how strongly easy examples are turned down
        # the gamma parameter controls the strength of the focal loss. A higher gamma value increases the down-weighting of easy examples, 
        # making the model focus more on hard examples. In this case, a gamma value of 2.0 is used, which is a common choice for focal loss.
        
        self.pos_weight = pos_weight # a one-element tensor that multiplies the loss on scam examples
        # the pos_weight parameter is used to give more importance to the positive class (scam examples) in the loss calculation. 
        # This helps to address the class imbalance in the dataset, where scam examples are much rarer than legitimate examples. 
        # By assigning a higher weight to the positive class, the model is encouraged to pay more attention to correctly classifying scam examples.
        
        self.reduction = reduction # 'mean', 'sum' or 'none' 
        # the reduction parameter specifies how the loss should be aggregated across the batch. 
        # It can take one of three values: 'mean', 'sum', or 'none'. If set to 'mean', the loss will be averaged over the batch; 
        # if set to 'sum', the loss will be summed; and if set to 'none', the loss will be returned for each 
        # individual example in the batch without any aggregation.

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits( # weight bce loss by pos_weight, which is a one-element tensor that multiplies the loss on scam examples.
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits) # sigmoid activation function is applied to the logits to convert them into probabilities, 
        # which are then used to calculate the focal loss. sigmoid since it returns 1 to 0, which is the probability of being a scam or not. 
        # The output of the sigmoid function is used to compute the focal loss.
        
        pt = torch.where(targets == 1, probs, 1 - probs) # model is correct and right if near 0 and wrong if near 1, 
        # so the focal term is (1 - pt) ** gamma.
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss