# -------------------------------------------------------------------------------------------------------------
# Created by: KK Nakka
# Email: krishna.nakka@epfl.ch
# Copyright (c) 2020
# Acknowledgement: Adapted from ProtoPNet codebase

# Usage

# python train_robust.py --net=AttProto --mode=robust --backbone=vgg16 --dataset=cub200

# ---------------------------------------------------------------------------------------------------------------


import os
import shutil
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datetime
import torchvision
import argparse
import re
import train_and_test_adv as tnt
import matplotlib.pyplot as plt
import pprint
import yaml


from utils import save
from utils.log import create_logger
from datasets.preprocess import mean, std, preprocess_input_function
from utils.helpers import makedir
from models import model_AttProto
from pushutils import push
from utils import save
from torch.backends import cudnn
cudnn.benchmark = True
from pprint import pformat
from utils.save import plot


# -----------------------------------------------------------------------------------------------------------


class Options():

    def __init__(self):
        parser = argparse.ArgumentParser(description='Train Robust Models')
        parser.add_argument('--backbone', type=str,
                            help='[vgg16|vgg19|resnet34')
        parser.add_argument('--dataset', type=str,
                            help='[cub200|cars196')
        parser.add_argument('--net', type=str,
                            help='[AttProto')
        parser.add_argument('--mode', type=str,
                            help='[normal|robust')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

# -----------------------------------------------------------------------------------------------------------


def main(args):

    with open("./configs/{}/{}_{}_{}.yaml".format(args.net, args.dataset, args.backbone,
                                                  args.mode)) as fp:
        cfg = yaml.safe_load(fp)

    NET_ARGS = cfg['NET_ARGS']
    DATA_ARGS = cfg['DATA_ARGS']
    EXP_ARGS = cfg['EXP_ARGS']
    class_specific = True

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model_dir = os.path.join('./saved_models/', args.dataset, args.backbone,
                             args.net, args.mode)

    makedir(model_dir)
    EXP_ARGS['model_dir'] = model_dir

    log, logclose = create_logger(log_filename=os.path.join(
        model_dir, 'train_logger_{}.txt'.format(datetime.datetime.now().strftime("%H:%M:%S"))))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'
    log(pformat(cfg))

    # ------------------------------- Get DataLoaders -----------------------------------------------------------

    normalize = transforms.Normalize(mean=NET_ARGS['mean'], std=NET_ARGS['std'])
    train_transforms = transforms.Compose([transforms.Resize(size=(
        DATA_ARGS['img_size'], DATA_ARGS['img_size'])), transforms.ToTensor(), ])   # removed normalize from here for adv training

    train_dataset = datasets.ImageFolder(DATA_ARGS['train_dir'], train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=EXP_ARGS['train_batch_size'], shuffle=True, num_workers=4, pin_memory=False)

    train_push_dataset = datasets.ImageFolder(DATA_ARGS['train_push_dir'], transforms.Compose([transforms.Resize(size=(
        DATA_ARGS['img_size'], DATA_ARGS['img_size'])), transforms.ToTensor(), ]))  # removed normalize from here for adv training
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=EXP_ARGS['train_push_batch_size'], shuffle=False, num_workers=4, pin_memory=False)

    test_dataset = datasets.ImageFolder(DATA_ARGS['test_dir'], transforms.Compose([transforms.Resize(size=(
        DATA_ARGS['img_size'], DATA_ARGS['img_size'])), transforms.ToTensor(), ]))  # removed normalize from here for adv training
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=EXP_ARGS['test_batch_size'], shuffle=False, num_workers=4, pin_memory=False)

    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(EXP_ARGS['train_batch_size']))

    # ---------------------------- Nodel and Optimizer -----------------------------------

    ppnet = model.construct_PPNet(base_architecture=NET_ARGS['base_architecture'],
                                  pretrained=True, img_size=DATA_ARGS['img_size'],
                                  prototype_shape=NET_ARGS['prototype_shape'],
                                  num_classes=DATA_ARGS['num_classes'],
                                  prototype_activation_function=NET_ARGS['prototype_activation_function'],
                                  add_on_layers_type=NET_ARGS['add_on_layers_type'],
                                  att_version=NET_ARGS['ATT_VERSION'])

    ppnet = ppnet.cuda()

    if EXP_ARGS['RESUME']['iS_RESUME']:
        ppnet = torch.load(EXP_ARGS['RESUME']['PATH'])
        log(" Resumed from model: {}".format(EXP_ARGS['RESUME']['PATH']))
        ppnet_multi = torch.nn.DataParallel(ppnet)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=True, log=log, EXP_ARGS=EXP_ARGS)
        log("\nInit Accuracy {:.2f} \n\n".format(accu))

    ppnet_multi = torch.nn.DataParallel(ppnet)

    warm_optimizer_lrs = EXP_ARGS['OPTIMIZER']['warm_optimizer_lrs']
    warm_optimizer_specs = [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
                            {'params': ppnet.prototype_vectors,
                                'lr': warm_optimizer_lrs['prototype_vectors']},
                            {'params': ppnet.att_layer.parameters(), 'lr': warm_optimizer_lrs['att_layer'], 'weight_decay': 1e-3}]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    joint_optimizer_lrs = EXP_ARGS['OPTIMIZER']['joint_optimizer_lrs']
    joint_optimizer_specs = [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
                             {'params': ppnet.add_on_layers.parameters(
                             ), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
                             {'params': ppnet.prototype_vectors,
                                 'lr': joint_optimizer_lrs['prototype_vectors']},
                             {'params': ppnet.att_layer.parameters(), 'lr': joint_optimizer_lrs['att_layer'], 'weight_decay': 1e-3}]

    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        joint_optimizer, step_size=int(joint_optimizer_lrs['joint_lr_step_size']), gamma=0.1)
    push_epochs = [i for i in range(EXP_ARGS['num_train_epochs']) if i % 10 == 0]

    log('\n\n--------------------------------Start Training ---------------------------\n\n')

    max_acc = 0.0
    max_acc_epoch = 0
    max_acc_iter = 0
    target_accu = 0.00

    import copy

    for epoch in range(EXP_ARGS['start_epoch'], EXP_ARGS['num_train_epochs']):

        log('--------------------------------------  Epoch: {}  --------------------------------------'.format(
            epoch))

        if epoch < EXP_ARGS['num_warm_epochs']:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                          class_specific=class_specific, coefs=EXP_ARGS['LOSS']['loss_coefs_warm'], log=log, EXP_ARGS=EXP_ARGS)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                          class_specific=class_specific, coefs=EXP_ARGS['LOSS']['loss_coefs_joint'], log=log, EXP_ARGS=EXP_ARGS)

        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log, EXP_ARGS=EXP_ARGS)

        if accu > max_acc:
            max_acc = accu
            max_acc_iter = 0
            max_acc_epoch = epoch
            save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                        model_name='', accu=accu,
                                        target_accu=target_accu, log=log, best=True,
                                        stage='prepush_{}'.format(epoch),
                                        num_classes=DATA_ARGS['num_classes'])

        log("\nBest Accuracy {:.2f} at epoch {} and iter {}\n\n".format(
            max_acc, max_acc_epoch, max_acc_iter))

        if epoch >= EXP_ARGS['push_start'] and epoch in push_epochs:

            log('\n--------------------------------Push Prototypes --------------------------')

            push.push_prototypes(train_push_loader,
                                 prototype_network_parallel=ppnet_multi,
                                 class_specific=class_specific,
                                 preprocess_input_function=preprocess_input_function,
                                 prototype_layer_stride=1,
                                 root_dir_for_saving_prototypes=img_dir, epoch_number=epoch,
                                 prototype_img_filename_prefix=prototype_img_filename_prefix,
                                 prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                 proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                                 save_prototype_class_identity=True, log=log)

            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log, EXP_ARGS=EXP_ARGS)

            save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                        model_name='',
                                        accu=accu, target_accu=target_accu,
                                        log=log, best=True, stage='push_{}'.format(epoch),
                                        num_classes=DATA_ARGS['num_classes'])

            last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(
            ), 'lr': EXP_ARGS['OPTIMIZER']['last_layer_optimizer_lrs']['last_layer_optimizer_lr']}]
            last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
            last_lr_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                last_layer_optimizer, step_size=EXP_ARGS['OPTIMIZER']['last_layer_optimizer_lrs']
                ['last_lr_step_size'], gamma=0.1)

            log('\n------------------------ Last Layer Training ---------------------------------')

            if NET_ARGS['prototype_activation_function'] != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)

                max_acc_post, max_acc_post_iter, max_acc_post_epoch = 0, 0, epoch

                for i in range(EXP_ARGS['OPTIMIZER']['last_layer_optimizer_lrs']['last_layer_optimizer_iters']):
                    log('Last layer optimization, Iteration:  {0}'.format(i))
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                  class_specific=class_specific, coefs=EXP_ARGS['LOSS']['loss_coefs_joint'],
                                  log=log, EXP_ARGS=EXP_ARGS)
                    last_lr_lr_scheduler.step()
                    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log, EXP_ARGS=EXP_ARGS)

                    if accu > max_acc_post:
                        max_acc_post = accu
                        max_acc_post_iter = i
                        max_acc_post_epoch = epoch
                        save.save_model_w_condition(model=ppnet,
                                                    model_dir=model_dir,
                                                    model_name='',
                                                    accu=accu,
                                                    target_accu=target_accu, log=log,
                                                    best=True, stage='postpush_{}'.format(epoch),
                                                    num_classes=DATA_ARGS['num_classes'])

                    plot(ppnet, model_dir + "/tsneplots/", str(epoch) +
                         '_' + str(i) + 'push', DATA_ARGS['num_classes'])

                    log("Best Accuracy - PostPush {:.2f} at epoch {} and iter {}"
                        .format(max_acc_post, max_acc_post_epoch, max_acc_post_iter))

                save.save_model_w_condition(model=ppnet,
                                            model_dir=model_dir, model_name='',
                                            accu=accu, target_accu=target_accu,
                                            log=log, best=True,
                                            stage='postpushfinal_{}'.format(epoch))

    logclose()


if __name__ == '__main__':

    args = Options().parse()
    main(args)
