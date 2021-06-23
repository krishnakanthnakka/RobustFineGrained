import os
import torch
from utils import *


def plot(ppnet_multi, model_dir, suffix, num_classes):

    prototype_vectors = ppnet_multi.prototype_vectors.cpu().detach().numpy()
    prototype_class_identity = torch.max(ppnet_multi.prototype_class_identity, dim=1)[1]
    prototype_class_identity = prototype_class_identity.cpu().detach().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex='col', sharey='row',
                           gridspec_kw={'hspace': 0, 'wspace': 0})
    prototype_vectors = np.squeeze(prototype_vectors)

    if True:
        plot_TSNE(prototype_vectors, prototype_class_identity, num_classes, 0, ax=ax, save_dir="./")
        fig.savefig(osp.join(model_dir, 'proto_{}.png'.format(suffix)),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)


def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print, best=False, stage=None, best_accu_stage=None, num_classes=None):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        #torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))

        if best is True:
            # if accu> best_accu_stage:
            torch.save(obj=model, f=os.path.join(model_dir, (stage + '.pth')))
        else:
            torch.save(obj=model, f=os.path.join(
                model_dir, (model_name + '{0:.4f}.pth').format(accu)))

    #plot(model, model_dir+"/tsneplots/", model_name, num_classes)
