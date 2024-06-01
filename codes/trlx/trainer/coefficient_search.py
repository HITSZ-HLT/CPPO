import torch
from torch.nn.parameter import Parameter
import torch.optim as optim
from time import time
import trlx.utils.logging as logging
logger = logging.get_logger(__name__)


def get_mean(coef, idxs, device):
    return torch.gather(coef, dim=-1, index=torch.LongTensor(idxs).to(device)).mean().to(device)


def train_couple_coef( R, P, loss_fn, end_cond, config):
    time_coefs = time()
    device = R.device
    N = R.shape[0]
    meta = Parameter(torch.randn([N, 2])).to(device)

    optimizer = optim.Adam([
        {'params': meta, 'lr': config.method.coefs_lr},
    ])
    for i in range(config.method.num_steps):
        optimizer.zero_grad()
        loss, coefs = loss_fn(meta, R, P, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(meta, 1.0)
        optimizer.step()
        stop = end_cond(coefs, R, P, config)

        if stop == 1.0:
            time_coefs = time() - time_coefs
            logger.info(f"[Coef Learning] early stop(={stop:.2}) at {i}-step, time:{time_coefs:.4}")
            break
    if stop!=1.0:
        time_coefs = time() - time_coefs
        logger.info(f"[Coef Learning] NO early stop(={stop:.2}) at {i}-step, time:{time_coefs:.4}")
    return coefs.detach(),i


def coef_loss_fn(meta, R, P, config):
    N=R.shape[0]
    coefs = (config.method.ub-config.method.lb) * torch.sigmoid(meta) + config.method.lb
    device = R.device
    coff_learn = coefs[:, 0]
    coff_reg = coefs[:, 1]

    rewards_std = R.std()
    rewards_mean = R.mean()
    logP_mean = P.mean()
    logP_std = P.std()

    threhold11 = (rewards_mean + config.method.threhold * rewards_std)
    threhold12 = (rewards_mean - config.method.threhold * rewards_std)
    threhold21 = (logP_mean + config.method.threhold * logP_std)
    threhold22 = (logP_mean - config.method.threhold * logP_std)

    cond11 = R > threhold11
    cond12 = R < threhold12
    cond21 = P > threhold21
    cond22 = P < threhold22
    cases = [[] for _ in range(5)]

    for i in range(N):
        if cond21[i] and cond11[i]:  # case 1: high high
            cases[0].append(i)
        elif cond21[i] and cond12[i]:  # case 2: high low
            cases[1].append(i)
        elif cond22[i] and cond11[i]:  # case 3: low high
            cases[2].append(i)
        elif cond22[i] and cond12[i]:  # case 4: low low
            cases[3].append(i)
        else:  # case 5: others
            cases[4].append(i)

    # var_loss = torch.stack([coff_learn * P, coff_learn * R]).var(dim=0).mean()

    if config.method.loss_type == "v1": # 当前最佳
        h_loss = config.method.expectation_coef * ((coff_learn - 1).square().mean()) + config.method.expectation_coef * (
            (coff_reg - 1).square().mean()) + coff_learn.var() + coff_reg.var()
    elif config.method.loss_type == "v2":
        h_loss = config.method.expectation_coef * ((coff_learn - 1).square().mean()) + config.method.expectation_coef * (
            (coff_reg - 1).square().mean())
    elif config.method.loss_type == "v3":
        h_loss = config.method.expectation_coef * ((coff_learn - 1).mean().square() ) + config.method.expectation_coef * (
                (coff_reg - 1).mean().square() ) + coff_learn.var() + coff_learn.var()
    elif config.method.loss_type == "v4":
        h_loss = config.method.expectation_coef * ((coff_learn - 1).mean().square()) + config.method.expectation_coef * (
            (coff_reg - 1).mean().square()) + coff_learn.var() + coff_learn.var() + torch.stack(
            [coff_learn * P, coff_learn * R]).var(dim=0).mean()


    # h_loss += expectation_coef * var_loss

    if len(cases[4]) != 0:
        learn_others = get_mean(coff_learn, cases[4], device)
        reg_others = get_mean(coff_reg, cases[4], device)
    if len(cases[0]) != 0:
        learn_case1 = get_mean(coff_learn, cases[0], device)
        reg_case1 = get_mean(coff_reg, cases[0], device)
    if len(cases[1]) != 0:
        learn_case2 = get_mean(coff_learn, cases[1], device)
        reg_case2 = get_mean(coff_reg, cases[1], device)
    if len(cases[2]) != 0:
        learn_case3 = get_mean(coff_learn, cases[2], device)
        reg_case3 = get_mean(coff_reg, cases[2], device)
    if len(cases[3]) != 0:
        learn_case4 = get_mean(coff_learn, cases[3], device)
        reg_case4 = get_mean(coff_reg, cases[3], device)

    if len(cases[0]) != 0 and len(cases[4]) != 0:
        h_loss += (learn_others - learn_case1) + (reg_others - reg_case1)
    if len(cases[1]) != 0 and len(cases[4]) != 0:
        h_loss += learn_others - learn_case2
    if len(cases[2]) != 0 and len(cases[4]) != 0:
        h_loss += (learn_others - learn_case3) - (reg_others - reg_case3)
    if len(cases[3]) != 0 and len(cases[4]) != 0:
        h_loss += -(learn_others - learn_case4) - (reg_others - reg_case4)

    return h_loss, coefs


def end_cond(coefs, R, P, config):
    N=R.shape[0]
    device = R.device
    coff_learn = coefs[:, 0]
    coff_reg = coefs[:, 1]

    rewards_std = R.std()
    rewards_mean = R.mean()
    logP_mean = P.mean()
    logP_std = P.std()

    threhold11 = (rewards_mean + config.method.threhold * rewards_std)
    threhold12 = (rewards_mean - config.method.threhold * rewards_std)
    threhold21 = (logP_mean + config.method.threhold * logP_std)
    threhold22 = (logP_mean - config.method.threhold * logP_std)

    cond11 = R > threhold11
    cond12 = R < threhold12
    cond21 = P > threhold21
    cond22 = P < threhold22
    cases = [[] for _ in range(5)]

    for i in range(N):
        if cond21[i] and cond11[i]:  # case 1: high high
            cases[0].append(i)
        elif cond21[i] and cond12[i]:  # case 2: high low
            cases[1].append(i)
        elif cond22[i] and cond11[i]:  # case 3: low high
            cases[2].append(i)
        elif cond22[i] and cond12[i]:  # case 4: low low
            cases[3].append(i)
        else:  # case 5: others
            cases[4].append(i)

    if len(cases[4]) != 0:
        learn_others = get_mean(coff_learn, cases[4], device)
        reg_others = get_mean(coff_reg, cases[4], device)
    if len(cases[0]) != 0:
        learn_case1 = get_mean(coff_learn, cases[0], device)
        reg_case1 = get_mean(coff_reg, cases[0], device)
    if len(cases[1]) != 0:
        learn_case2 = get_mean(coff_learn, cases[1], device)
        reg_case2 = get_mean(coff_reg, cases[1], device)
    if len(cases[2]) != 0:
        learn_case3 = get_mean(coff_learn, cases[2], device)
        reg_case3 = get_mean(coff_reg, cases[2], device)
    if len(cases[3]) != 0:
        learn_case4 = get_mean(coff_learn, cases[3], device)
        reg_case4 = get_mean(coff_reg, cases[3], device)

    float_cond = 0.0
    total = 2.0

    if (coff_reg.mean() - 1).abs() < 0.2:
        float_cond += 1.
    if (coff_learn.mean() - 1).abs() < 0.2:
        float_cond += 1.

    if len(cases[0]) != 0 and len(cases[4]) != 0:
        total += 2.0
        if (learn_others < learn_case1):
            float_cond += 1.
        if (reg_others < reg_case1):
            float_cond += 1.
    if len(cases[1]) != 0 and len(cases[4]) != 0:
        total += 1.0
        if learn_others < learn_case2:
            float_cond += 1
    if len(cases[2]) != 0 and len(cases[4]) != 0:
        total += 2.0
        if (learn_others < learn_case3):
            float_cond += 1
        if (reg_others > reg_case3):
            float_cond += 1
    if len(cases[3]) != 0 and len(cases[4]) != 0:
        total += 2.0
        if (learn_others > learn_case4):
            float_cond += 1
        if (reg_others > reg_case4):
            float_cond += 1

    return float_cond / total