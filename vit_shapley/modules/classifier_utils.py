import ipdb
import torch
from torch.nn import functional as F
from torchmetrics import Accuracy, MeanMetric, AUROC, Precision, F1Score, Recall, CohenKappa


def set_metrics(pl_module):
    for phase in ["train", "val", "test"]:
        if pl_module.hparams.target_type == "binary":
            setattr(pl_module, f"{phase}_loss", MeanMetric())

            setattr(pl_module, f"{phase}_accuracy", Accuracy())
            setattr(pl_module, f"{phase}_precision",
                    Precision(num_classes=pl_module.hparams.output_dim, average="macro"))
            setattr(pl_module, f"{phase}_recall",
                    Recall(num_classes=pl_module.hparams.output_dim, average="macro"))
            setattr(pl_module, f"{phase}_f1", F1Score(num_classes=pl_module.hparams.output_dim, average="macro"))
            setattr(pl_module, f"{phase}_cohenkappa",
                    CohenKappa(num_classes=2, weights='quadratic'))
            setattr(pl_module, f"{phase}_auroc", AUROC())

        elif pl_module.hparams.target_type == "multiclass":
            setattr(pl_module, f"{phase}_loss", MeanMetric())

            setattr(pl_module, f"{phase}_accuracy", Accuracy())
            setattr(pl_module, f"{phase}_precision",
                    Precision(num_classes=pl_module.hparams.output_dim, average="macro"))
            setattr(pl_module, f"{phase}_recall",
                    Recall(num_classes=pl_module.hparams.output_dim, average="macro"))
            setattr(pl_module, f"{phase}_f1", F1Score(num_classes=pl_module.hparams.output_dim, average="macro"))
            setattr(pl_module, f"{phase}_cohenkappa",
                    CohenKappa(num_classes=pl_module.hparams.output_dim, weights='quadratic'))

        elif "multilabel" in pl_module.hparams.target_type:
            output_select = pl_module.hparams.target_type.split('-')[1].replace('loss', '')
            if output_select == 'all':
                setattr(pl_module, f"{phase}_loss", MeanMetric())
            else:
                setattr(pl_module, f"{phase}_loss", MeanMetric())

            output_select = pl_module.hparams.target_type.split('-')[2].replace('metric', '')
            if output_select == 'all':
                pass
            else:
                setattr(pl_module, f"{phase}_accuracy", Accuracy())
                setattr(pl_module, f"{phase}_precision",
                        Precision(num_classes=1, average="macro"))
                setattr(pl_module, f"{phase}_recall",
                        Recall(num_classes=1, average="macro"))
                setattr(pl_module, f"{phase}_f1",
                        F1Score(num_classes=1, average="macro"))
                setattr(pl_module, f"{phase}_auroc", AUROC())
        else:
            NotImplementedError("Not supported target type. It should be one of binary, multiclass, multilabel")


def epoch_wrapup(pl_module, phase):
    if pl_module.hparams.target_type == "binary":
        loss = getattr(pl_module, f"{phase}_loss").compute()
        getattr(pl_module, f"{phase}_loss").reset()
        pl_module.log(f"{phase}/epoch_loss", loss)

        accuracy = getattr(pl_module, f"{phase}_accuracy").compute()
        getattr(pl_module, f"{phase}_accuracy").reset()
        pl_module.log(f"{phase}/epoch_accuracy", accuracy)

        precision = getattr(pl_module, f"{phase}_precision").compute()
        getattr(pl_module, f"{phase}_precision").reset()
        pl_module.log(f"{phase}/epoch_precision", precision)

        recall = getattr(pl_module, f"{phase}_recall").compute()
        getattr(pl_module, f"{phase}_recall").reset()
        pl_module.log(f"{phase}/epoch_recall", recall)

        f1 = getattr(pl_module, f"{phase}_f1").compute()
        getattr(pl_module, f"{phase}_f1").reset()
        pl_module.log(f"{phase}/epoch_f1", f1)

        cohenkappa = getattr(pl_module, f"{phase}_cohenkappa").compute()
        getattr(pl_module, f"{phase}_cohenkappa").reset()
        pl_module.log(f"{phase}/epoch_cohenkappa", cohenkappa)

        auroc = getattr(pl_module, f"{phase}_auroc").compute()
        getattr(pl_module, f"{phase}_auroc").reset()
        pl_module.log(f"{phase}/epoch_auroc", auroc)

    elif pl_module.hparams.target_type == "multiclass":
        loss = getattr(pl_module, f"{phase}_loss").compute()
        getattr(pl_module, f"{phase}_loss").reset()
        pl_module.log(f"{phase}/epoch_loss", loss)

        accuracy = getattr(pl_module, f"{phase}_accuracy").compute()
        getattr(pl_module, f"{phase}_accuracy").reset()
        pl_module.log(f"{phase}/epoch_accuracy", accuracy)

        precision = getattr(pl_module, f"{phase}_precision").compute()
        getattr(pl_module, f"{phase}_precision").reset()
        pl_module.log(f"{phase}/epoch_precision", precision)

        recall = getattr(pl_module, f"{phase}_recall").compute()
        getattr(pl_module, f"{phase}_recall").reset()
        pl_module.log(f"{phase}/epoch_recall", recall)

        f1 = getattr(pl_module, f"{phase}_f1").compute()
        getattr(pl_module, f"{phase}_f1").reset()
        pl_module.log(f"{phase}/epoch_f1", f1)

        cohenkappa = getattr(pl_module, f"{phase}_cohenkappa").compute()
        getattr(pl_module, f"{phase}_cohenkappa").reset()
        pl_module.log(f"{phase}/epoch_cohenkappa", cohenkappa)

    elif "multilabel" in pl_module.hparams.target_type:
        output_select = pl_module.hparams.target_type.split('-')[1].replace('loss', '')
        if output_select == 'all':
            loss = getattr(pl_module, f"{phase}_loss").compute()
            getattr(pl_module, f"{phase}_loss").reset()
            pl_module.log(f"{phase}/epoch_loss", loss)
        else:
            loss = getattr(pl_module, f"{phase}_loss").compute()
            getattr(pl_module, f"{phase}_loss").reset()
            pl_module.log(f"{phase}/epoch_loss", loss)

        output_select = pl_module.hparams.target_type.split('-')[2].replace('metric', '')
        if output_select == 'all':
            pass
        else:
            accuracy = getattr(pl_module, f"{phase}_accuracy").compute()
            getattr(pl_module, f"{phase}_accuracy").reset()
            pl_module.log(f"{phase}/epoch_accuracy", accuracy)

            precision = getattr(pl_module, f"{phase}_precision").compute()
            getattr(pl_module, f"{phase}_precision").reset()
            pl_module.log(f"{phase}/epoch_precision", precision)

            recall = getattr(pl_module, f"{phase}_recall").compute()
            getattr(pl_module, f"{phase}_recall").reset()
            pl_module.log(f"{phase}/epoch_recall", recall)

            f1 = getattr(pl_module, f"{phase}_f1").compute()
            getattr(pl_module, f"{phase}_f1").reset()
            pl_module.log(f"{phase}/epoch_f1", f1)

            auroc = getattr(pl_module, f"{phase}_auroc").compute()
            getattr(pl_module, f"{phase}_auroc").reset()
            pl_module.log(f"{phase}/epoch_auroc", auroc)
    else:
        NotImplementedError("Not supported target type. It should be one of binary, multiclass, multilabel")

    if pl_module.hparams.checkpoint_metric == "CohenKappa":
        checkpoint_metric = cohenkappa
    elif pl_module.hparams.checkpoint_metric == "AUC":
        checkpoint_metric = auroc
    elif pl_module.hparams.checkpoint_metric == "accuracy":
        checkpoint_metric = accuracy
    else:
        raise NotImplementedError("Not supported checkpoint metric")
    pl_module.log(f"{phase}/checkpoint_metric", checkpoint_metric)


def compute_metrics(pl_module, logits, labels, phase):
    # phase = "train" if pl_module.training else "val"
    if pl_module.hparams.target_type == "binary":
        if pl_module.hparams.loss_weight is None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())  # internally computes sigmoid
        else:
            loss = pl_module.hparams.loss_weight[0] / (pl_module.hparams.loss_weight[0] + pl_module.hparams.loss_weight[
                1]) * F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=torch.tensor(
                pl_module.hparams.loss_weight[1] / pl_module.hparams.loss_weight[0]).float())
            F.binary_cross_entropy_with_logits(logits, labels.float(), weight=torch.tensor(5).float())
        loss = getattr(pl_module, f"{phase}_loss")(loss)
        pl_module.log(f"{phase}/loss", loss)

        accuracy = getattr(pl_module, f"{phase}_accuracy")(torch.sigmoid(logits), labels)
        pl_module.log(f"{phase}/accuracy", accuracy)

        recall = getattr(pl_module, f"{phase}_recall")(torch.sigmoid(logits), labels)
        pl_module.log(f"{phase}/recall", recall)

        precision = getattr(pl_module, f"{phase}_precision")(torch.sigmoid(logits), labels)
        pl_module.log(f"{phase}/precision", precision)

        f1 = getattr(pl_module, f"{phase}_f1")(torch.sigmoid(logits), labels)
        pl_module.log(f"{phase}/f1", f1)

        cohenkappa = getattr(pl_module, f"{phase}_cohenkappa")(torch.sigmoid(logits), labels)
        pl_module.log(f"{phase}/cohenkappa", cohenkappa)

        auroc = getattr(pl_module, f"{phase}_auroc")(torch.sigmoid(logits), labels)
        pl_module.log(f"{phase}/auroc", auroc)

    elif pl_module.hparams.target_type == "multiclass":
        loss = F.cross_entropy(logits, labels)  # internally computes softmax
        loss = getattr(pl_module, f"{phase}_loss")(loss)
        pl_module.log(f"{phase}/loss", loss)

        accuracy = getattr(pl_module, f"{phase}_accuracy")(torch.softmax(logits, dim=1), labels)
        pl_module.log(f"{phase}/accuracy", accuracy)

        recall = getattr(pl_module, f"{phase}_recall")(torch.softmax(logits, dim=1), labels)
        pl_module.log(f"{phase}/recall", recall)

        precision = getattr(pl_module, f"{phase}_precision")(torch.softmax(logits, dim=1), labels)
        pl_module.log(f"{phase}/precision", precision)

        f1 = getattr(pl_module, f"{phase}_f1")(torch.softmax(logits, dim=1), labels)
        pl_module.log(f"{phase}/f1", f1)

        cohenkappa = getattr(pl_module, f"{phase}_cohenkappa")(torch.softmax(logits, dim=1), labels)
        pl_module.log(f"{phase}/cohenkappa", cohenkappa)

    elif "multilabel" in pl_module.hparams.target_type:
        output_select = pl_module.hparams.target_type.split('-')[1].replace('loss', '')
        if output_select == 'all':
            loss = F.binary_cross_entropy_with_logits(logits, labels.type_as(logits))  # internally computes sigmoid
            loss = getattr(pl_module, f"{phase}_loss")(loss)
            pl_module.log(f"{phase}/loss", loss)
        else:
            loss = F.binary_cross_entropy_with_logits(logits[:, int(output_select)],
                                                      labels[:, int(output_select)].type_as(
                                                          logits))  # internally computes sigmoid
            loss = getattr(pl_module, f"{phase}_loss")(loss)
            pl_module.log(f"{phase}/loss", loss)

        output_select = pl_module.hparams.target_type.split('-')[2].replace('metric', '')
        if output_select == 'all':
            pass
        else:
            accuracy = getattr(pl_module, f"{phase}_accuracy")(torch.sigmoid(logits[:, int(output_select)]),
                                                               labels[:, int(output_select)])
            pl_module.log(f"{phase}/accuracy", accuracy)

            precision = getattr(pl_module, f"{phase}_precision")(torch.sigmoid(logits[:, int(output_select)]),
                                                                 labels[:, int(output_select)])
            pl_module.log(f"{phase}/precision", precision)

            recall = getattr(pl_module, f"{phase}_recall")(torch.sigmoid(logits[:, int(output_select)]),
                                                           labels[:, int(output_select)])
            pl_module.log(f"{phase}/recall", recall)

            f1 = getattr(pl_module, f"{phase}_f1")(torch.sigmoid(logits[:, int(output_select)]),
                                                   labels[:, int(output_select)])
            pl_module.log(f"{phase}/f1", f1)

            auroc = getattr(pl_module, f"{phase}_auroc")(torch.sigmoid(logits[:, int(output_select)]),
                                                         labels[:, int(output_select)])
            pl_module.log(f"{phase}/auroc", auroc)

    else:
        NotImplementedError("Not supported target type. It should be one of binary, multiclass, multilabel")

    return loss
