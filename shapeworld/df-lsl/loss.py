import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

def cal_pred_loss(hint_model,hint_seq,hint_length,image_rep,label,multimodal_model, scorer_model,examples_rep, infer_hyp, multimodal_concept,poe):
    hint_rep = hint_model(hint_seq, hint_length)
    # Use concept to compute prediction loss
    # (how well does example repr match image repr)?
    score = scorer_model.score(examples_rep, image_rep)
    pred_loss = F.binary_cross_entropy_with_logits(
        score, label.float())
    score_hint = scorer_model.score(hint_rep, image_rep)
    pred_loss += F.binary_cross_entropy_with_logits(
        score_hint, label.float())
    score_true = scorer_model.score(hint_rep, examples_rep)
    pred_loss += F.binary_cross_entropy_with_logits(
        score_true, torch.ones(label.size()).cuda())
    return pred_loss

def cal_hypo_loss(train_vocab_size, batch_size, n_ex, hint_seq,hint_length, examples_rep, hint_model, multimodal_model, proposal_model,use_hyp,predict_image_hyp,predict_hyp,predict_hyp_task,multimodal_concept,examples_rep_mean):
    if predict_image_hyp:
        # Use raw images, flatten out tasks
        hyp_batch_size = batch_size * n_ex
        hyp_source_rep = examples_rep.view(hyp_batch_size, -1)
        hint_seq = hint_seq.unsqueeze(1).repeat(1, n_ex, 1).view(
            hyp_batch_size, -1)
        hint_length = hint_length.unsqueeze(1).repeat(
            1, n_ex).view(hyp_batch_size)
    else:
        hyp_source_rep = examples_rep_mean
        hyp_batch_size = batch_size

    if predict_hyp and predict_hyp_task == 'embed':
        # Encode hints, minimize distance between hint and images/examples
        hint_rep = hint_model(hint_seq, hint_length)
        if multimodal_concept:
            hint_rep = multimodal_model(hint_rep, examples_rep_mean)
        dists = torch.norm(hyp_source_rep - hint_rep, p=2, dim=1)
        hypo_loss = torch.mean(dists)
    else:
        # Decode images/examples to hints
        hypo_out = proposal_model(hyp_source_rep, hint_seq,
                                    hint_length)
        seq_len = hint_seq.size(1)
        hypo_out = hypo_out[:, :-1].contiguous()
        hint_seq = hint_seq[:, 1:].contiguous()

        hypo_out_2d = hypo_out.view(hyp_batch_size * (seq_len - 1),
                                    train_vocab_size)
        hint_seq_2d = hint_seq.long().view(hyp_batch_size * (seq_len - 1))
        hypo_loss = F.cross_entropy(hypo_out_2d,
                                    hint_seq_2d,
                                    reduction='none')
        hypo_loss = hypo_loss.view(hyp_batch_size, (seq_len - 1))
        hypo_loss = torch.mean(torch.sum(hypo_loss, dim=1))
    return hypo_loss


def cal_accuracy(batch_size,image_rep,sos_index,eos_index,pad_index,hint_seq,hint_length,device,label_np,examples_rep,scorer_model,proposal_model,multimodal_model,hint_model,poe,infer_hyp,n_infer,oracle,multimodal_concept):
    score = 0
    if infer_hyp:
        
        for num in range(examples_rep.size(1)):
            for j in range(n_infer):
                # Decode greedily for first hyp; otherwise sample
                # If --oracle, hint_seq/hint_length is given
                hint_seq = hint_seq.to(device)
                hint_length = hint_length.to(device)
                hint_rep = hint_model(hint_seq, hint_length)
                if multimodal_concept:
                    hint_rep = multimodal_model(
                        hint_rep, examples_rep[:,num,:])
                score += scorer_model.score(hint_rep, image_rep)
                if poe:
                    image_score = scorer_model.score(examples_rep[:,num,:],image_rep)
                    # Average with image score
                    score += image_score
        score = score / float(examples_rep.size(1))
        label_hat = score > 0
        label_hat = label_hat.cpu().numpy()
            # Update scores and predictions for best running hints
        accuracy = accuracy_score(label_np, label_hat)
    else:
        # Compare image directly to example rep
        for num in range(examples_rep.size(1)):
            score += scorer_model.score(examples_rep[:,num,:], image_rep)
        score = score / float(examples_rep.size(1))
        label_hat = score > 0
        label_hat = label_hat.cpu().numpy()
        accuracy = accuracy_score(label_np, label_hat)
    return accuracy, label_hat