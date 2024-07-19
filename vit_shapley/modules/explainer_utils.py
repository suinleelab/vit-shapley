import torch
from torch.nn import functional as F
from torchmetrics import MeanMetric


def set_metrics(pl_module):
    for phase in ["train", "val", "test"]:
        setattr(pl_module, f"{phase}_value_diff", MeanMetric())
        setattr(pl_module, f"{phase}_efficiency_gap", MeanMetric())
        setattr(pl_module, f"{phase}_efficiency_class_gap", MeanMetric())
        setattr(pl_module, f"{phase}_loss", MeanMetric())


def epoch_wrapup(pl_module, phase):
    value_diff = getattr(pl_module, f"{phase}_value_diff").compute()
    getattr(pl_module, f"{phase}_value_diff").reset()
    pl_module.log(f"{phase}/epoch_value_diff", value_diff)

    efficiency_gap = getattr(pl_module, f"{phase}_efficiency_gap").compute()
    getattr(pl_module, f"{phase}_efficiency_gap").reset()
    pl_module.log(f"{phase}/epoch_efficiency_gap", efficiency_gap)

    efficiency_class_gap = getattr(pl_module, f"{phase}_efficiency_class_gap").compute()
    getattr(pl_module, f"{phase}_efficiency_class_gap").reset()
    pl_module.log(f"{phase}/epoch_efficiency_class_gap", efficiency_class_gap)

    loss = getattr(pl_module, f"{phase}_loss").compute()
    getattr(pl_module, f"{phase}_loss").reset()
    pl_module.log(f"{phase}/epoch_loss", loss)

    if pl_module.hparams.checkpoint_metric == "loss":
        checkpoint_metric = -loss
    elif pl_module.hparams.checkpoint_metric == "value_diff":
        checkpoint_metric = -value_diff
    else:
        raise NotImplementedError("Not supported checkpoint metric")
    pl_module.log(f"{phase}/checkpoint_metric", checkpoint_metric)


def compute_metrics(pl_module, num_players,
                    values_pred, values_target,
                    efficiency_lambda,
                    value_pred_beforenorm_sum, surrogate_grand, surrogate_null,
                    efficiency_class_lambda, value_pred_beforenorm_sum_class,
                    phase):
    value_diff = num_players * F.mse_loss(input=values_pred, target=values_target,
                                          reduction='mean')  # (batch, num_mask_samples, num_classes), (batch, num_mask_samples, num_classes)

    efficiency_gap = num_players * F.mse_loss(input=value_pred_beforenorm_sum, target=surrogate_grand - surrogate_null,
                                              reduction='mean')  # (batch, num_classes), (1, num_classes)

    efficiency_class_gap = F.mse_loss(input=value_pred_beforenorm_sum_class,
                                      target=torch.zeros_like(value_pred_beforenorm_sum_class),
                                      reduction='mean')  # (batch, num_players)

    loss = value_diff + efficiency_lambda * efficiency_gap + efficiency_class_lambda * efficiency_class_gap

    value_diff = getattr(pl_module, f"{phase}_value_diff")(value_diff)
    pl_module.log(f"{phase}/value_diff", value_diff)

    efficiency_gap = getattr(pl_module, f"{phase}_efficiency_gap")(efficiency_gap)
    pl_module.log(f"{phase}/efficiency_gap", efficiency_gap)

    efficiency_class_gap = getattr(pl_module, f"{phase}_efficiency_class_gap")(efficiency_class_gap)
    pl_module.log(f"{phase}/efficiency_class_gap", efficiency_class_gap)

    loss = getattr(pl_module, f"{phase}_loss")(loss)
    pl_module.log(f"{phase}/loss", loss)

    return loss
