import ipdb
import torch
from torch.nn import functional as F
from torchmetrics import MeanMetric


def set_metrics(pl_module):
    for phase in ["train", "val", "test"]:
        setattr(pl_module, f"{phase}_loss", MeanMetric())


def epoch_wrapup(pl_module, phase):
    loss = getattr(pl_module, f"{phase}_loss").compute()
    getattr(pl_module, f"{phase}_loss").reset()
    pl_module.log(f"{phase}/epoch_loss", loss)

    if pl_module.hparams.checkpoint_metric == "loss":
        checkpoint_metric = -loss
    else:
        raise NotImplementedError("Not supported checkpoint metric")
    pl_module.log(f"{phase}/checkpoint_metric", checkpoint_metric)


def compute_metrics(pl_module, logits, logits_target, phase):
    if pl_module.hparams.target_type == "binary":
        loss = F.kl_div(input=torch.concat([F.logsigmoid(logits), F.logsigmoid(-logits)], dim=1),
                        target=torch.concat([torch.sigmoid(logits_target), torch.sigmoid(-logits_target)], dim=1),
                        reduction="batchmean",
                        log_target=False)
        loss = getattr(pl_module, f"{phase}_loss")(loss)
    elif pl_module.hparams.target_type == "multiclass":
        loss = F.kl_div(input=torch.log_softmax(logits, dim=1),
                        target=torch.softmax(logits_target, dim=1),
                        reduction='batchmean',
                        log_target=False)
        loss = getattr(pl_module, f"{phase}_loss")(loss)
    else:
        NotImplementedError("Not supported target type. It should be one of binary, multiclass, multilabel")

    pl_module.log(f"{phase}/loss", loss)

    return loss
