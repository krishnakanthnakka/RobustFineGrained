import time
import torch
from utils.helpers import list_of_distances, make_one_hot


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, EXP_ARGS=None):

    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_correct_att = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cross_entropy_att = 0
    total_avg_separation_cost = 0
    total_cluster_att_cost = 0
    total_separation_att_cost = 0

    prototype_vectors = torch.squeeze(model.module.prototype_vectors)

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        grad_req = torch.enable_grad() if is_train else torch.no_grad()

        with grad_req:

            output, min_distances, att_logits, distances, ep = model(input)
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            cross_entropy_att = torch.nn.functional.cross_entropy(att_logits, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                prototypes_of_correct_class = torch.t(
                    model.module.prototype_class_identity[:, label]).cuda()

                cm_att = ep['classmix_att']
                prototypes_of_correct_class_resized = prototypes_of_correct_class.unsqueeze(
                    2).unsqueeze(3)

                if EXP_ARGS['LOSS']['CLUSTER_ATT_COST']:
                    cluster_att_cost = torch.mean(
                        cm_att * (max_dist - torch.max((max_dist - distances)
                                                       * prototypes_of_correct_class_resized, dim=1)[0]))

                if EXP_ARGS['LOSS']['SEP_ATT_COST']:
                    prototypes_of_wrong_class_resized = 1 - prototypes_of_correct_class_resized
                    separation_att_cost = torch.mean(
                        cm_att * (max_dist - torch.max((max_dist - distances)
                                                       * prototypes_of_wrong_class_resized, dim=1)[0]))

                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            _, predicted_att = torch.max(att_logits.data, 1)
            n_correct_att += (predicted_att == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cross_entropy_att += cross_entropy_att.item()

            if EXP_ARGS['LOSS']['CLUSTER_ATT_COST']:
                total_cluster_att_cost += cluster_att_cost.item()

            if EXP_ARGS['LOSS']['SEP_ATT_COST']:
                total_separation_att_cost += separation_att_cost.item()

        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy +
                            coefs['l1'] * l1 + cross_entropy_att * 1.0)

                    if EXP_ARGS['LOSS']['CLUSTER_ATT_COST']:
                        loss += coefs['clst_att'] * cluster_att_cost

                    if EXP_ARGS['LOSS']['SEP_ATT_COST']:
                        loss += coefs['sep_att'] * separation_att_cost

            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

        if is_train and (i + 1) % 99 == 0:
            log('\n')
            log('\t   {:20s} \t{:<4d}'.format("Batch: ", i))
            log('\t   {:20s} \t{:<.5f}'.format("cross ent: ", total_cross_entropy / n_batches))
            log('\t   {:20s} \t{:<.5f}'.format(
                "cross ent_att: ", total_cross_entropy_att / n_batches))
            if EXP_ARGS['LOSS']['CLUSTER_ATT_COST']:
                log('\t   {:20s} \t{:<.10f}'.format(
                    "cluster_att: ", total_cluster_att_cost / n_batches))
            if EXP_ARGS['LOSS']['SEP_ATT_COST']:
                log('\t   {:20s} \t{:<.10f}'.format(
                    "separation_att: ", total_separation_att_cost / n_batches))
            log('\t   {:20s} \t{:.2f}%'.format("accu: ", n_correct / n_examples * 100))
            log('\t   {:20s} \t{:.2f}%'.format(
                "accu_att: ", n_correct_att / n_examples * 100))

    end = time.time()

    log('\n')
    log('\t{:20s}\t{:<.2f}'.format("time: ", end - start))
    log('\t{:20s}\t{:<6d}'.format("num of examples: ", n_examples))
    log('\t{:20s} \t{:<.5f}'.format("cross ent: ", total_cross_entropy / n_batches))
    log('\t{:20s} \t{:<.5f}'.format(
        "cross ent_att: ", total_cross_entropy_att / n_batches))
    if EXP_ARGS['LOSS']['CLUSTER_ATT_COST']:
        log('\t{:20s} \t{:<.10f}'.format(
            "cluster_att: ", total_cluster_att_cost / n_batches))
    if EXP_ARGS['LOSS']['SEP_ATT_COST']:
        log('\t{:20s} \t{:<.10f}'.format(
            "separation_att: ", total_separation_att_cost / n_batches))
    log('\t{:20s} \t{:.2f}%'.format("accu: ", n_correct / n_examples * 100))
    log('\t{:20s} \t{:.2f}%'.format(
        "accu_att: ", n_correct_att / n_examples * 100))

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print, EXP_ARGS=None):
    assert(optimizer is not None)

    log('\nMode: Train Data')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, EXP_ARGS=EXP_ARGS)


def test(model, dataloader, class_specific=False, log=print, EXP_ARGS=None):
    log('\nMode: Test Data')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log, EXP_ARGS=EXP_ARGS)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    for p in model.module.att_layer.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\nOptimizer: Last Layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    for p in model.module.att_layer.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\nOptimizer: Warm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    for p in model.module.att_layer.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('Optimizer: Joint')
