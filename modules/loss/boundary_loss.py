import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label using scatter_ to ensure device consistency"""

    # 1. Ensure label is LongTensor and add the channel dimension
    # Input Shape: [Batch, Height, Width] -> [2, 40, 1800]
    # Unsqueezed:  [Batch, 1, Height, Width] -> [2, 1, 40, 1800]
    label = label.long().unsqueeze(1)

    # 2. explicit device handling
    device = label.device

    # 3. Create a zeros tensor directly on the correct device
    # Target Shape: [Batch, Classes, Height, Width] -> [2, 14, 40, 1800]
    shape = (label.shape[0], n_classes, label.shape[2], label.shape[3])
    one_hot_label = torch.zeros(shape, device=device, dtype=torch.float32)

    # 4. Scatter ones into the correct indices
    # dim=1 corresponds to the 'n_classes' dimension
    one_hot_label.scatter_(1, label, 1.0)

    if requires_grad:
        one_hot_label.requires_grad_(True)

    return one_hot_label


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        # pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth

        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # Visualization Boundary
        # for i in range(c):
        #     gt_bv = gt_b.detach().cpu().numpy()
        #     # cv2.imshow('gt_b_cls.png'.format(i), gt_bv[0][i])
        #     cv2.imwrite('gt_b_cls{}.png'.format(i), gt_bv[0][i]*255)
        #
        #     pred_bv = pred_b.detach().cpu().numpy()
        #     #cv2.imshow('pred_b_cls{}'.format(i), pred_bv[0][i])
        #     cv2.imwrite('pred_b_cls{}.png'.format(i), pred_bv[0][i]*255)

        # reshape
        #         gt_b = gt_b[:, 1:, :, :]
        #         pred_b = pred_b[:, 1:, :, :]
        #         c = c-1

        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss
