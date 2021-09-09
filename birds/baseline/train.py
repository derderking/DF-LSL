"""
Train script.
"""

import json
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm

import constants
from data import lang_utils
from data.datamgr import SetDataManager
from io_utils import get_resume_file, model_dict, parse_args
from models.language import TextProposal, TextRep
from models.protonet import ProtoNet


def get_optimizer(model, args):
    """
    Get the optimizer for the model based on arguments. Specifically, if
    needed, we split up training into (1) main parameters, (2) RNN-specific
    parameters, with different learning rates if specified.

    :param model: nn.Module to train
    :param args: argparse.Namespace - other args passed to the script

    :returns: a torch.optim.Optimizer
    """
    # Get params
    main_params = {"params": []}
    rnn_params = {"params": [], "lr": args.rnn_lr_scale * args.lr}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("language_model."):
            # Scale RNN learning rate
            rnn_params["params"].append(param)
        else:
            main_params["params"].append(param)
    if args.lsl and not rnn_params["params"]:
        print("Warning: --lsl is set but no RNN parameters found")
    params_to_optimize = [main_params, rnn_params]

    # Define optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
    elif args.optimizer == "amsgrad":
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, amsgrad=True)
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(params_to_optimize, lr=args.lr)
    else:
        raise NotImplementedError("optimizer = {}".format(args.optimizer))
    return optimizer


def train(
    base_loader,
    val_loader,
    model,
    start_epoch,
    stop_epoch,
    args,
    metrics_fname="metrics.json",
):
    """
    Main training script.

    :param base_loader: torch.utils.data.DataLoader for training set, generated
        by data.datamgr.SetDataManager
    :param val_loader: torch.utils.data.DataLoader for validation set,
        generated by data.datamgr.SetDataManager
    :param model: nn.Module to train
    :param start_epoch: which epoch we started at
    :param stop_epoch: which epoch to end at
    :param args: other arguments passed to the script
    "param metrics_fname": where to save metrics
    """
    optimizer = get_optimizer(model, args)

    max_val_acc = 0
    best_epoch = 0

    val_accs = []
    val_losses = []
    all_metrics = defaultdict(list)
    for epoch in tqdm(
        range(start_epoch, stop_epoch), total=stop_epoch - start_epoch, desc="Train"
    ):
        model.train()
        metric = model.train_loop(epoch, base_loader, optimizer, args)
        for m, val in metric.items():
            all_metrics[m].append(val)
        model.eval()

        os.makedirs(args.checkpoint_dir, exist_ok=True)

        val_acc, val_loss = model.test_loop(val_loader,)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        if val_acc > max_val_acc:
            best_epoch = epoch
            tqdm.write("best model! save...")
            max_val_acc = val_acc
            outfile = os.path.join(args.checkpoint_dir, "best_model.tar")
            torch.save({"epoch": epoch, "state": model.state_dict()}, outfile)

        if epoch and (epoch % args.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(args.checkpoint_dir, "{:d}.tar".format(epoch))
            torch.save({"epoch": epoch, "state": model.state_dict()}, outfile)
        tqdm.write("")

        # Save metrics
        metrics = {
            "train_acc": all_metrics["train_acc"],
            "current_train_acc": all_metrics["train_acc"][-1],
            "train_loss": all_metrics["train_loss"],
            "current_train_loss": all_metrics["train_loss"][-1],
            "cls_loss": all_metrics["cls_loss"],
            "current_cls_loss": all_metrics["cls_loss"][-1],
            "lang_loss": all_metrics["lang_loss"],
            "current_lang_loss": all_metrics["lang_loss"][-1],
            "current_epoch": epoch,
            "val_acc": val_accs,
            "val_loss": val_losses,
            "current_val_loss": val_losses[-1],
            "current_val_acc": val_acc,
            "best_epoch": best_epoch,
            "best_val_acc": max_val_acc,
        }
        with open(os.path.join(args.checkpoint_dir, metrics_fname), "w") as fout:
            json.dump(metrics, fout, sort_keys=True, indent=4, separators=(",", ": "))

        # Save a copy to current metrics too
        if (
            metrics_fname != "metrics.json"
            and metrics_fname.startswith("metrics_")
            and metrics_fname.endswith(".json")
        ):
            metrics["n"] = int(metrics_fname[8])
            with open(os.path.join(args.checkpoint_dir, "metrics.json"), "w") as fout:
                json.dump(
                    metrics, fout, sort_keys=True, indent=4, separators=(",", ": ")
                )

    # If didn't train, save model anyways
    if stop_epoch == 0:
        outfile = os.path.join(args.checkpoint_dir, "best_model.tar")
        torch.save({"epoch": stop_epoch, "state": model.state_dict()}, outfile)


if __name__ == "__main__":
    args = parse_args("train")
    
    torch.cuda.set_device(0)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
    # I don't seed the np rng since dataset loading uses multiprocessing with
    # random choices.
    # https://github.com/numpy/numpy/issues/9650
    # Unavoidable undeterminism here for now

    base_file = os.path.join(constants.DATA_DIR, "base.json")
    val_file = os.path.join(constants.DATA_DIR, "val.json")

    # Load language
    vocab = lang_utils.load_vocab(constants.LANG_DIR)

    l3_model = None
    lang_model = None
    if args.lsl or args.l3:
        if args.glove_init:
            vecs = lang_utils.glove_init(vocab, emb_size=args.lang_emb_size)
        embedding_model = nn.Embedding(
            len(vocab), args.lang_emb_size, _weight=vecs if args.glove_init else None
        )
        if args.freeze_emb:
            embedding_model.weight.requires_grad = False

        lang_input_size = 1600
        lang_model = TextProposal(
            embedding_model,
            input_size=lang_input_size,
            hidden_size=args.lang_hidden_size,
            project_input=lang_input_size != args.lang_hidden_size,
            rnn=args.rnn_type,
            num_layers=args.rnn_num_layers,
            dropout=args.rnn_dropout,
            vocab=vocab,
            **lang_utils.get_special_indices(vocab)
        )

        if args.l3:
            l3_model = TextRep(
                embedding_model,
                hidden_size=args.lang_hidden_size,
                rnn=args.rnn_type,
                num_layers=args.rnn_num_layers,
                dropout=args.rnn_dropout,
            )
            l3_model = l3_model.cuda()

        embedding_model = embedding_model.cuda()
        lang_model = lang_model.cuda()

    # if test_n_way is smaller than train_n_way, reduce n_query to keep batch
    # size small
    n_query = max(1, int(16 * args.test_n_way / args.train_n_way))

    train_few_shot_args = dict(n_way=args.train_n_way, n_support=args.n_shot)
    base_datamgr = SetDataManager(
        "CUB", 84, n_query=n_query, **train_few_shot_args, args=args
    )
    print("Loading train data")

    base_loader = base_datamgr.get_data_loader(
        base_file,
        aug=True,
        lang_dir=constants.LANG_DIR,
        normalize=True,
        vocab=vocab,
        # Maximum training data restrictions only apply at train time
        max_class=args.max_class,
        max_img_per_class=args.max_img_per_class,
        max_lang_per_class=args.max_lang_per_class,
    )

    val_datamgr = SetDataManager(
        "CUB",
        84,
        n_query=n_query,
        n_way=args.test_n_way,
        n_support=args.n_shot,
        args=args,
    )
    print("Loading val data\n")
    val_loader = val_datamgr.get_data_loader(
        val_file, aug=False, lang_dir=constants.LANG_DIR, normalize=True, vocab=vocab,
    )
    # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

    model = ProtoNet(
        model_dict[args.model],
        **train_few_shot_args,
        # Language options
        lsl=args.lsl,
        language_model=lang_model,
        lang_supervision=args.lang_supervision,
        l3=args.l3,
        l3_model=l3_model,
        l3_n_infer=args.l3_n_infer
    )

    model = model.cuda()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    start_epoch = args.start_epoch
    stop_epoch = args.stop_epoch

    if args.resume:
        resume_file = get_resume_file(args.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp["epoch"] + 1
            model.load_state_dict(tmp["state"])

    metrics_fname = "metrics_{}.json".format(args.n)

    train(
        base_loader,
        val_loader,
        model,
        start_epoch,
        stop_epoch,
        args,
        metrics_fname=metrics_fname,
    )
