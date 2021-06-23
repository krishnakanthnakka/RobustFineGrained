import time
import torch
from utils.helpers import list_of_distances, make_one_hot
from datasets.preprocess import mean, std
from utils.adv_utils import attack_fns_AttProto, get_attack_params


def normalize_fn(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, EXPS_ARGS=None):

    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_correct_att = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cross_entropy_att = 0
    total_cluster_att_cost = 0
    total_separation_att_cost = 0

    FEAT_LOSS = False

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()
        criterion = torch.nn.CrossEntropyLoss().cuda()

        if True:

            ATTACK_ARGS = EXPS_ARGS['ATTACK_ARGS'] if is_train else EXPS_ARGS['ATTACK_EVAL_ARGS']
            adv = attack_fns_AttProto(model, criterion, input, target,
                                      eps=float(ATTACK_ARGS['EPS']),
                                      alpha=float(ATTACK_ARGS['ALPHA']),
                                      attack_type=ATTACK_ARGS['TYPE'],
                                      iters=ATTACK_ARGS['ITERS'],
                                      normalize_fn=normalize_fn,
                                      branch='FR')

            adv_labels = target

            # Concat both normal and adversarial images
            if False:
                input = torch.cat((input, adv), 0)
                target = torch.cat((target, adv_labels))
                label = torch.cat((label, label))
                num_clean = int(input.shape[0] / 2)

            else:
                input = adv
                target = adv_labels

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        prototype_vectors = torch.squeeze(model.module.prototype_vectors)

        with torch.enable_grad():

            output, min_distances, att_logits, distances, ep = model(
                normalize_fn(input.clone().detach()))

            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            cross_entropy_att = torch.nn.functional.cross_entropy(att_logits, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                prototypes_of_correct_class = torch.t(
                    model.module.prototype_class_identity[:, label]).cuda()

                if EXPS_ARGS['LOSS']['CLUSTER_ATT_COST']:

                    cm_att = ep['classmix_att']
                    cluster_att_cost = torch.mean(cm_att * torch.min(distances, dim=1)[0])

                    prototypes_of_correct_class_resized = prototypes_of_correct_class.unsqueeze(
                        2).unsqueeze(3)
                    cluster_att_cost = torch.mean(
                        cm_att * (max_dist - torch.max((max_dist - distances)
                                                       * prototypes_of_correct_class_resized, dim=1)[0]))

                if EXPS_ARGS['LOSS']['SEP_ATT_COST']:
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

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            _, predicted_att = torch.max(att_logits.data, 1)
            n_correct_att += (predicted_att == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cross_entropy_att += cross_entropy_att.item()

            if EXPS_ARGS['LOSS']['CLUSTER_ATT_COST']:

                total_cluster_att_cost += cluster_att_cost.item()

            if EXPS_ARGS['LOSS']['SEP_ATT_COST']:
                total_separation_att_cost += separation_att_cost.item()

        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy +
                            coefs['l1'] * l1 + cross_entropy_att * 1.0)

                    if EXPS_ARGS['LOSS']['CLUSTER_ATT_COST']:
                        loss += coefs['clst_att'] * cluster_att_cost

                    if EXPS_ARGS['LOSS']['SEP_ATT_COST']:
                        loss += coefs['sep_att'] * separation_att_cost

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances
        del att_logits
        del distances
        del ep

        if is_train and (i + 1) % 99 == 0:

            log('\n')
            log('\t   {:20s} \t{:<4d}'.format("Batch: ", i))
            log('\t   {:20s} \t{:<.5f}'.format("cross ent: ", total_cross_entropy / n_batches))
            log('\t   {:20s} \t{:<.5f}'.format(
                "cross ent_att: ", total_cross_entropy_att / n_batches))
            if EXPS_ARGS['LOSS']['CLUSTER_ATT_COST']:
                log('\t   {:20s} \t{:<.10f}'.format(
                    "cluster_att: ", total_cluster_att_cost / n_batches))
            if EXPS_ARGS['LOSS']['SEP_ATT_COST']:
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
    if EXPS_ARGS['LOSS']['CLUSTER_ATT_COST']:
        log('\t{:20s} \t{:<.10f}'.format(
            "cluster_att: ", total_cluster_att_cost / n_batches))
    if EXPS_ARGS['LOSS']['SEP_ATT_COST']:
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
                          class_specific=class_specific, coefs=coefs, log=log, EXPS_ARGS=EXP_ARGS)


def test(model, dataloader, class_specific=False, log=print, EXP_ARGS=None):
    log('\nMode: Test Data')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log, EXPS_ARGS=EXP_ARGS)


def plot(ppnet_multi, model_dir):

    prototype_vectors = ppnet_multi.module.prototype_vectors.cpu().detach().numpy()
    prototype_class_identity = torch.max(ppnet_multi.module.prototype_class_identity, dim=1)[1]
    prototype_class_identity = prototype_class_identity.cpu().detach().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex='col', sharey='row',
                           gridspec_kw={'hspace': 0, 'wspace': 0})
    prototype_vectors = np.squeeze(prototype_vectors)

    if True:
        plot_TSNE(prototype_vectors, prototype_class_identity, 12, 0, ax=ax, save_dir="./")
        fig.savefig(osp.join(model_dir, 'proto.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)


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
