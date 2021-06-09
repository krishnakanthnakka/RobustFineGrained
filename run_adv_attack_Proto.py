# -------------------------------------------------------------------------------------------------------------
# Created by: KK Nakka
# Email: krishna.nakka@epfl.ch
# Copyright (c) 2020
# Acknowledgement: Adapted from ProtoPNet codebase

# python run_adv_attack_Proto.py -dataset=cub200  -config=settings_robust.yaml -mode=robust -split=test -backbone=vgg16 -net=Proto -checkpoint=model.pth -attack=pgd10_2

# ---------------------------------------------------------------------------------------------------------------

import os
import argparse
import yaml
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import warnings
warnings.filterwarnings("ignore")
from models import model_Proto
from utils import normalize_fn
from utils.adv_utils import attack_fns_Proto, get_attack_params

# ---------------------------------------------------------------------------------------------------------------


class Options():

    def __init__(self):
        parser = argparse.ArgumentParser(description='Run Adversarial Attacks')
        parser.add_argument('-gpu_id', type=str, default='0')
        parser.add_argument('-backbone', type=str,
                            help='[vgg16|vgg19|resnet34')
        parser.add_argument('-dataset', type=str,
                            help='[cub200|cars196')
        parser.add_argument('-net', type=str,
                            help='Proto')
        parser.add_argument('-mode', type=str,
                            help='[normal|robust')
        parser.add_argument('-checkpoint', type=str,
                            help='saved model name')
        parser.add_argument('-split', type=str,
                            default='test')
        parser.add_argument('-config', type=str,
                            help='settings_normal.yaml|settings_robust_yaml. \
                            Please place this along with model.pth in the saved models directory')
        parser.add_argument('-attack', type=str,
                            help='[fgsm1_2|fgsm1_8|bim10_2|bim10_8|pgd10_2|pgd10_8|mim10_2|mim10_8]')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        #print("\n", args)
        return args


# ---------------------------------------------------------------------------------------------------------------


def main(args):

    with open("./saved_models/{}/{}/{}/{}/{}".format(args.dataset, args.backbone, args.net, args.mode, args.config)) as f:
        cfg = yaml.safe_load(f)

    # ---------------------------------------------Prepare configuration for experiment -------------------------
    NET_ARGS = cfg['NET_ARGS']
    DATA_ARGS = cfg['DATA_ARGS']
    EXP_ARGS = cfg['EXP_ARGS']
    attack_dir = DATA_ARGS['attack_test_dir'] if args.split == 'test' else DATA_ARGS['attack_train_dir']

    # --------------------------------------------- Prepare the dataset------------------------------------------
    attack_dataset = datasets.ImageFolder(attack_dir, transforms.Compose([transforms.Resize(
        size=(DATA_ARGS['img_size'], DATA_ARGS['img_size'])), transforms.ToTensor(), ]))

    data_loader = torch.utils.data.DataLoader(attack_dataset,
                                              batch_size=EXP_ARGS['train_push_batch_size'],
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=False,
                                              sampler=None)
    total_batches = len(data_loader)

    # --------------------------------------------- Prepare model -----------------------------------------------
    modelpath = './saved_models/{}/{}/{}/{}/{}'.format(args.dataset,
                                                       args.backbone, args.net, args.mode, args.checkpoint)

    # (Commented) Load full model instead of state_dict.
    # But gives error when the code is refactored to merge multiple folders.
    # Dont use torch.save() with full object in  the future.

    #ppnet = torch.load(modelpath)
    #torch.save(ppnet.state_dict(), 'model.pth')

    # -------------------  Construct model and load state_dict --------------------------------------------------

    ppnet = model_Proto.construct_PPNet(base_architecture=NET_ARGS['base_architecture'],
                                        pretrained=False,
                                        img_size=DATA_ARGS['img_size'],
                                        prototype_shape=NET_ARGS['prototype_shape'],
                                        num_classes=DATA_ARGS['num_classes'],
                                        prototype_activation_function=NET_ARGS['prototype_activation_function'],
                                        add_on_layers_type=NET_ARGS['add_on_layers_type'],
                                        )

    ppnet.load_state_dict(torch.load(modelpath))

    ppnet_multi = torch.nn.DataParallel(ppnet)
    ppnet_multi.eval()

    print("\nModel path is {}".format(modelpath))

    correct_fr = 0
    correct_att = 0
    num_samples = 0
    adv_correct_fr = 0
    adv_correct_att = 0

    criterion = torch.nn.CrossEntropyLoss().cuda()

    # --------------------------------------------- Prepare attack ----------------------------------------------

    attack_params = get_attack_params(args.attack)
    print("Attack type: {}, Epsilon: {:.5f}, Alpha: {:.5f},  Iters: {}\n".format(
        attack_params['TYPE'], attack_params['EPS'], attack_params['ALPHA'], attack_params['ITERS']))

    # --------------------------------------------- Run attack --------------------------------------------------

    for it, (data, target) in enumerate(data_loader):

        data, target = data.cuda(), target.cuda()
        num_samples += data.shape[0]

        # ------------------------- predictions on clean image --------------------------------------------------
        output_fr = ppnet_multi(
            normalize_fn(data.clone().detach()))[0]

        pred_fr = output_fr.data.max(1, keepdim=True)[1]

        correct_fr += pred_fr.eq(target.data.view_as(pred_fr)).cpu().sum()

        # -------------------------prepare adversarial image-------------------------
        adv_img = attack_fns_Proto(ppnet_multi, criterion, data, target, eps=attack_params['EPS'],
                                   alpha=attack_params['ALPHA'], attack_type=attack_params['TYPE'],
                                   iters=attack_params['ITERS'], normalize_fn=normalize_fn)

        # -------------------------predictions on adversarial image----------------------------------------------
        output_adv_fr = ppnet_multi(
            normalize_fn(adv_img.clone().detach()))[0]

        pred_adv_fr = output_adv_fr.data.max(1, keepdim=True)[1]
        adv_correct_fr += torch.sum(output_adv_fr.argmax(dim=-1) ==
                                    target).item()

        # -----------------------------Print Stats---------------------------------------------------------------
        print("Batch: [{}/{}]\t FR_branch Normal: {:.2f}%\t"
              "FR_branch Adv: {: .2f}%, "
              .format(it, total_batches,
                      (100.0 * float(correct_fr) / num_samples),
                      (100.0 * float(adv_correct_fr) / num_samples)))
        del data, target, output_fr, output_adv_fr

    print("Final \t  FR_branch Normal: {:.2f}%\t"
          "FR_branch Adv: {:.2f}%\n"
          .format(
              (100.0 * float(correct_fr) / num_samples),
              (100.0 * float(adv_correct_fr) / num_samples)))


if __name__ == '__main__':

    args = Options().parse()
    main(args)
