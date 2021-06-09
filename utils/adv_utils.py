import torch
import numpy as np
import json


def get_attack_params(attacktype):

    ATTACK_MIM8 = {'TYPE': 'mim', 'ITERS': 10,
                   'EPS': 8.0 / 255.0, 'ALPHA': 0.8 / 255.0}
    ATTACK_MIM2 = {'TYPE': 'mim', 'ITERS': 10,
                   'EPS': 2.0 / 255.0, 'ALPHA': 0.2 / 255.0}

    ATTACK_BIM8 = {'TYPE': 'bim', 'ITERS': 10,
                   'EPS': 8.0 / 255.0, 'ALPHA': 0.8 / 255.0}
    ATTACK_BIM2 = {'TYPE': 'bim', 'ITERS': 10,
                   'EPS': 2.0 / 255.0, 'ALPHA': 0.2 / 255.0}

    ATTACK_FGSM8 = {'TYPE': 'fgsm', 'ITERS': 1,
                    'EPS': 8.0 / 255.0, 'ALPHA': 8.0 / 255.0}
    ATTACK_FGSM2 = {'TYPE': 'fgsm', 'ITERS': 1,
                    'EPS': 2.0 / 255.0, 'ALPHA': 2.0 / 255.0}

    ATTACK_PGD8 = {'TYPE': 'pgd', 'ITERS': 10,
                   'EPS': 8.0 / 255.0, 'ALPHA': 2.0 / 255.0}
    ATTACK_PGD2 = {'TYPE': 'pgd', 'ITERS': 10,
                   'EPS': 2.0 / 255.0, 'ALPHA': 1.0 / 255.0}

    ATTACKS = {'pgd10_2': ATTACK_PGD2, 'pgd10_8': ATTACK_PGD8, 'bim10_2': ATTACK_BIM2, 'bim10_8': ATTACK_BIM8,
               'fgsm1_8': ATTACK_FGSM8, 'fgsm1_2': ATTACK_FGSM2, 'mim10_8': ATTACK_MIM8, 'mim10_2': ATTACK_MIM2}

    return ATTACKS[attacktype]


def attack_fns_AttProto(model, criterion, img, label, eps, alpha, attack_type, iters, normalize_fn, branch):

    if attack_type == 'fgsm':
        delta = torch.zeros_like(img).uniform_(-eps, eps).cuda()
        delta.requires_grad = True

        output, _, output_att, distances, ep = model(
            normalize_fn(img + delta))[:5]
        loss = 0

        if branch == 'FR':
            loss += criterion(output, label)
            loss += criterion(output_att, label)
        elif branch == 'A':
            loss += criterion(output_att, label)
        else:
            assert False, "No Branch found with given configuration"
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -eps, eps)
        delta.data = torch.max(torch.min(1 - img, delta.data), 0 - img)
        delta = delta.detach()
        adv = torch.clamp(img + delta, 0, 1)
        return adv.detach()

    elif attack_type == 'bim':

        adv = img.detach()
        adv.requires_grad = True
        noise = 0
        for j in range(iters):
            output, _, output_att, distances, ep = model(
                normalize_fn(adv.clone()))[:5]
            loss = 0
            if branch == 'FR':
                loss += criterion(output, label)
                loss += criterion(output_att, label)
            elif branch == 'A':
                loss += criterion(output_att, label)
            else:
                assert False, "No Branch found with given configuration"
            loss.backward()
            noise = adv.grad
            adv.data = adv.data + alpha * noise.sign()
            adv.data.clamp_(0.0, 1.0)
            adv.grad.data.zero_()
        return adv.detach()

    elif attack_type == 'mim':
        adv = img.detach()
        adv.requires_grad = True
        noise = 0
        for j in range(iters):
            output, _, output_att, distances, ep = model(
                normalize_fn(adv.clone()))[:5]
            loss = 0
            if branch == 'FR':
                loss += criterion(output, label)
                loss += criterion(output_att, label)
            elif branch == 'A':
                loss += criterion(output_att, label)
            else:
                assert False, "No Branch found with given configuration"
            loss.backward()
            adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
            adv.data = adv.data + alpha * noise.sign()
            adv.data.clamp_(0.0, 1.0)
            adv.grad.data.zero_()
        return adv.detach()

    elif attack_type == 'pgd':
        delta = torch.zeros_like(img).uniform_(-eps, eps)
        delta.data = torch.max(torch.min(1 - img, delta.data), 0 - img)
        for _ in range(iters):
            delta.requires_grad = True
            output, _, output_att, distances, ep = model(
                normalize_fn(img + delta))[:5]
            loss = 0
            if branch == 'FR':
                loss += criterion(output, label)
                loss += criterion(output_att, label)
            elif branch == 'A':
                loss += criterion(output_att, label)
            else:
                assert False, "No Branch found with given configuration"

            loss.backward()
            grad = delta.grad.detach()
            I = output.max(1)[1] == label
            delta.data = torch.clamp(
                delta + alpha * torch.sign(grad), -eps, eps)
            delta.data = torch.max(torch.min(1 - img, delta.data), 0 - img)
            delta.grad.data.zero_()
        delta = delta.detach()
        adv = torch.clamp(img + delta, 0, 1)
        return adv.detach()

    else:
        assert False, "Exception: No attack impllementation found! Please check."


def attack_fns_AP(model, criterion, img, label, eps, alpha, attack_type, iters, normalize_fn):

    if attack_type == 'fgsm':
        delta = torch.zeros_like(img).uniform_(-eps, eps).cuda()
        delta.requires_grad = True
        output = model(normalize_fn(img + delta))[0]
        loss = criterion(output, label)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -eps, eps)
        delta.data = torch.max(torch.min(1 - img, delta.data), 0 - img)
        delta = delta.detach()
        adv = torch.clamp(img + delta, 0, 1)
        return adv.detach()

    elif attack_type == 'bim':

        adv = img.detach()
        adv.requires_grad = True
        noise = 0

        for j in range(iters):
            out_adv = model(normalize_fn(adv.clone()))[0]
            loss = criterion(out_adv, label)
            loss.backward()
            noise = adv.grad
            adv.data = adv.data + alpha * noise.sign()
            adv.data.clamp_(0.0, 1.0)
            adv.grad.data.zero_()

        return adv.detach()

    elif attack_type == 'mim':
        adv = img.detach()
        adv.requires_grad = True
        noise = 0

        for j in range(iters):

            out_adv = model(normalize_fn(adv.clone()))[0]
            loss = criterion(out_adv, label)
            loss.backward()

            adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
            adv.data = adv.data + alpha * noise.sign()
            adv.data.clamp_(0.0, 1.0)

            adv.grad.data.zero_()

        return adv.detach()

    elif attack_type == 'pgd':

        delta = torch.zeros_like(img).uniform_(-eps, eps)
        delta.data = torch.max(torch.min(1 - img, delta.data), 0 - img)
        for _ in range(iters):
            delta.requires_grad = True
            output = model(normalize_fn(img + delta))[0]
            loss = criterion(output, label)
            loss.backward()
            grad = delta.grad.detach()
            I = output.max(1)[1] == label
            delta.data = torch.clamp(
                delta + alpha * torch.sign(grad), -eps, eps)
            delta.data = torch.max(torch.min(1 - img, delta.data), 0 - img)
            # delta.data[I] = torch.clamp(delta + alpha * torch.sign(grad), -eps, eps)[I]   # TODO: Check if early stopping is needed or not
            # delta.data[I] = torch.max(torch.min(1-img, delta.data), 0-img)[I]
            delta.grad.data.zero_()

        delta = delta.detach()
        adv = torch.clamp(img + delta, 0, 1)
        return adv.detach()

    else:
        assert False, "Exception: No attack impllementation found! Please check."


def attack_fns_Proto(model, criterion, img, label, eps, alpha, attack_type, iters, normalize_fn):

    if attack_type == 'fgsm':
        delta = torch.zeros_like(img).uniform_(-eps, eps).cuda()
        delta.requires_grad = True
        output = model(normalize_fn(img + delta))[0]
        loss = criterion(output, label)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -eps, eps)
        delta.data = torch.max(torch.min(1 - img, delta.data), 0 - img)
        delta = delta.detach()
        adv = torch.clamp(img + delta, 0, 1)
        return adv.detach()

    elif attack_type == 'bim':

        adv = img.detach()
        adv.requires_grad = True
        noise = 0

        for j in range(iters):
            out_adv = model(normalize_fn(adv.clone()))[0]
            loss = criterion(out_adv, label)
            loss.backward()
            noise = adv.grad
            adv.data = adv.data + alpha * noise.sign()
            adv.data.clamp_(0.0, 1.0)
            adv.grad.data.zero_()

        return adv.detach()

    elif attack_type == 'mim':
        adv = img.detach()
        adv.requires_grad = True
        noise = 0

        for j in range(iters):

            out_adv = model(normalize_fn(adv.clone()))[0]
            loss = criterion(out_adv, label)
            loss.backward()

            adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
            adv.data = adv.data + alpha * noise.sign()
            adv.data.clamp_(0.0, 1.0)
            adv.grad.data.zero_()
        return adv.detach()

    elif attack_type == 'pgd':
        delta = torch.zeros_like(img).uniform_(-eps, eps)
        delta.data = torch.max(torch.min(1 - img, delta.data), 0 - img)
        for _ in range(iters):
            delta.requires_grad = True
            output = model(normalize_fn(img + delta))[0]
            loss = 0
            loss = criterion(output, label)
            loss.backward()
            grad = delta.grad.detach()
            I = output.max(1)[1] == label
            delta.data = torch.clamp(
                delta + alpha * torch.sign(grad), -eps, eps)
            delta.data = torch.max(torch.min(1 - img, delta.data), 0 - img)
            # delta.data[I] = torch.clamp(delta + alpha * torch.sign(grad), -eps, eps)[I]   # TODO: Check if early stopping is needed or not
            # delta.data[I] = torch.max(torch.min(1-img, delta.data), 0-img)[I]
            delta.grad.data.zero_()
        delta = delta.detach()
        adv = torch.clamp(img + delta, 0, 1)
        return adv.detach()

    else:
        assert False, "Exception: No attack impllementation found! Please check."
