import argparse
import logging
import os
import re
from metrics import cal_dice,cal_hd95
import random
import shutil
import sys
from loss import CombinedLoss
import time
#from xml.etree.ElementInclude import default_loader
from dataloaders.utils_brats import get_data_loader
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
#from tensorboardX import SummaryWriter
# from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# from torch.distributions import Categorical
# from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
# import augmentations
# from PIL import Image

# from dataloaders import utils
# from dataloaders.dataset import (
#     BaseDataSets,
#     CTATransform,
#     RandomGenerator,
#     TwoStreamBatchSampler,
#     WeakStrongAugment,
# )
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps, util
from val_2D import test_single_volume, test_single_volume_fast

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, default="BMC", help="experiment_name")
parser.add_argument("--model", type=str, default="unet_3D", help="model_name")
parser.add_argument("--max_iterations", type=int,
                    default=12000, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=2,
                    help="batch_size per gpu")
parser.add_argument("--deterministic", type=int, default=1,
                    help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.03,
                    help="segmentation network learning rate")
parser.add_argument("--patch_size", type=list,
                    default=[128, 128], help="patch size of network input")
parser.add_argument("--seed", type=int, default=2023, help="random seed")
parser.add_argument("--num_classes", type=int, default=4,
                    help="output channel of network")
parser.add_argument("--load", default=False,
                    action="store_true", help="restore previous checkpoint")
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.8,
    help="confidence threshold for using pseudo-labels",
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory_3d(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    # db_train = BaseDataSets(
    #     base_dir=args.root_path,
    #     split="train",
    #     transform=RandomGenerator(args.patch_size)
    # )
    # db_val = BaseDataSets(base_dir=args.root_path, split="val")
    source_root = 'D:/HDU/STORE/BRATS_dataloader/source'
    target_root = 'D:/HDU/STORE/BRATS_dataloader/target'
    
    trainloader,valloader = get_data_loader(
        source_root=source_root,
        target_root=target_root,
        train_path='train',
        test_path='test',
        batch_train=args.batch_size,
        batch_test=args.batch_size,
        nw = 0,#linux可修改为>0
        img=img,
        mode=mode
    )

    model = create_model()
    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
    #                          num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    # valloader = DataLoader(db_val, batch_size=1, shuffle=False,
    #                        num_workers=1)

    model.train()

    #ce_loss = CrossEntropyLoss()
    #dice_loss = losses.DiceLoss(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor([0.05, 2.0, 0.1, 0.5], device=device)
    dice_reduction='macro'
    #数量顺序 0,2,3,1
    criterion = CombinedLoss(
        ce_weight=2.0,
        dice_weight=3.0,
        dice_reduction=dice_reduction,
        class_weights=class_weights,
        device=device
    )

    ##writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = 200
    best_performance = 0.6

    iter_num = int(iter_num)

    #iterator = tqdm(range(start_epoch, max_epoch), ncols=120)
    best_dice = 0
    train_flag = True
    for epoch_num in range(max_epoch):
        if train_flag :
        
            train_bar = tqdm(trainloader, desc=f"Epoch {epoch_num + 1}/{max_epoch} Training")
            dice1_total = 0.0
            dice2_total = 0.0
            dice3_total = 0.0
            batch_count = 0
            for i_batch, sampled_batch in enumerate(train_bar):
                image_batch, label_batch = (
                    sampled_batch["image"],
                    sampled_batch["label"],
                )
                #print('111')
                #print('imgshape:',image_batch.shape)
                #print('labelshape:',label_batch.shape)
                label_batch = label_batch.squeeze(1)
                #print('imgshape:',image_batch.shape)
                #print('labelshape:',label_batch.shape)
                image_batch, label_batch = (
                    image_batch.cuda(),
                    label_batch.cuda(),
                )

                # model preds
                outputs = model(image_batch)
                outputs_soft = torch.softmax(outputs, dim=1)

                # loss = 0.5 * (ce_loss(outputs, label_batch.long(
                #     )) + dice_loss(outputs_soft[:, 1, ...], label_batch))
                loss,dice1, dice2, dice3 = criterion(outputs, label_batch)
                batch_count += 1
                dice1_total += dice1
                dice2_total += dice2
                dice3_total += dice3
                    # 计算历史平均dice值
                dice1_avg = dice1_total / batch_count
                dice2_avg = dice2_total / batch_count
                dice3_avg = dice3_total / batch_count

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_num = iter_num + 1

                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                dice_mean_avg = (dice1_avg + dice2_avg + dice3_avg) / 3
                train_bar.desc = f"Epoch [{epoch_num}/{max_epoch}] dice:[{dice_mean_avg:.4f}] loss:[{loss.item():.4f}] Training"
                #train_bar.desc = f"Epoch [{epoch_num}/{max_epoch}] loss:[{loss:.4f}] Training"
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_
            avg_dice1_train = dice1_total / len(trainloader)
            avg_dice2_train = dice2_total / len(trainloader)
            avg_dice3_train = dice3_total / len(trainloader)

                #writer.add_scalar("lr", lr_, iter_num)
                #writer.add_scalar("loss/model_loss", loss, iter_num)
                # logging.info("iteration %d : model loss : %f" %
                #              (iter_num, loss.item()))
                #if (iter_num + 1)  % 50 == 0:
                    # show weakly augmented image
                    #image = image_batch[1, 0:1, :, :]
                    #writer.add_image("train/Image", image, iter_num)

                    #outputs_strong = torch.argmax(
                        #outputs_soft, dim=1, keepdim=True)
                    #writer.add_image("train/model_Prediction",
                                    #outputs_strong[1, ...] * 50, iter_num)
                    # show ground truth label
                    #labs = label_batch[1, ...].unsqueeze(0) * 50
                    #writer.add_image("train/GroundTruth", labs, iter_num)

        if (epoch_num + 1) % 1  == 0:
                model.eval()
                #metric_list = 0.0
                dice_total = 0.0
                hd95_total = 0.0
                dice1_val = 0.0
                dice2_val = 0.0
                dice3_val = 0.0
                hd95_list_wt = []
                hd95_list_co = []
                hd95_list_ec = []
                batch_count = 0
                
                with torch.no_grad():
                    val_bar = tqdm(valloader, desc=f"Epoch {epoch_num + 1}/{max_epoch} Validation")
                    for i_batch, sampled_batch in enumerate(val_bar):
                        image_batch, label_batch = (
                            sampled_batch["image"],
                            sampled_batch["label"],
                        )
                        label_batch = label_batch.squeeze(1)
                        image_batch, label_batch = (
                            image_batch.cuda(),
                            label_batch.cuda(),
                        )
                        outputs = model(image_batch)
                        #outputs_soft = torch.softmax(outputs, dim=1)
                        dice1, dice2, dice3 = cal_dice(outputs, label_batch)
                        dice1 = dice1.item()
                        dice2 = dice2.item()
                        dice3 = dice3.item()
                        hd95_wt, hd95_co, hd95_ec = cal_hd95(outputs, label_batch)
                        batch_count += 1
                        dice_total += dice1+dice2+dice3
                        dice_mean_val = dice_total / 3 /batch_count
                        hd95_total += hd95_wt+hd95_co+hd95_ec
                        hd95_mean_val = hd95_total / 3 /batch_count
                        dice1_val += dice1
                        dice2_val += dice2
                        dice3_val += dice3
                        hd95_list_wt.append(hd95_wt)
                        hd95_list_co.append(hd95_co)
                        hd95_list_ec.append(hd95_ec)
                        val_bar.desc = f"Epoch [{epoch_num}/{max_epoch}] dice:[{dice_mean_val:.4f}] hd95:[{hd95_mean_val:.4f}] Validation"

                avg_dice1_val = dice1_val / len(valloader)
                avg_dice2_val = dice2_val / len(valloader)
                avg_dice3_val = dice3_val / len(valloader)
                avg_hd95_wt = np.nanmean(hd95_list_wt)
                avg_hd95_co = np.nanmean(hd95_list_co)
                avg_hd95_ec = np.nanmean(hd95_list_ec)
                avg_dice = (avg_dice1_val + avg_dice2_val + avg_dice3_val) / 3
                #writer.add_scalar("info/model_val_mean_dice",
                                  #performance, iter_num)
                if avg_dice > best_dice:
                    best_performance = avg_dice
                    save_mode_path = os.path.join(
                        snapshot_path,
                        "model_iter_{}_dice_{}.pth".format(
                            iter_num, round(best_performance, 4)),
                    )
                    save_best = os.path.join(
                        snapshot_path, "{}_best_model.pth".format(args.model))
                    util.save_checkpoint(
                        epoch_num, model, optimizer, loss, save_mode_path)
                    util.save_checkpoint(
                        epoch_num, model, optimizer, loss, save_best)

                logging.info(
                    "epoch %d : model_mean_dice: %f" % (
                        epoch_num, dice_mean_avg)
                )
                txt_file = os.path.join(snapshot_path, "val-{}.txt".format(img))
                with open(txt_file, 'a') as f:
                    f.write(f"Epoch: {epoch_num}/{max_epoch}\n")
                    
                    if train_flag:
                        f.write(f"Train Dice: ET {avg_dice1_train:.3f} TC {avg_dice2_train:.3f} WT {avg_dice3_train:.3f}\n")
                    f.write(f"Val Dice: ET {avg_dice1_val:.3f} TC {avg_dice2_val:.3f} WT {avg_dice3_val:.3f}\n")
                    f.write(f"Val HD95: ET {avg_hd95_wt:.3f} TC {avg_hd95_co:.3f} WT {avg_hd95_ec:.3f}\n\n")
        model.train()

        if (epoch_num+1) % 1 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "model_epoch_" + str(epoch_num) + ".pth")
                util.save_checkpoint(
                    epoch_num, model, optimizer, loss, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))



    #writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    # if os.path.exists(snapshot_path + "/code"):
    #     shutil.rmtree(snapshot_path + "/code")
    # shutil.copytree(".", snapshot_path + "/code",
    #                 shutil.ignore_patterns([".git", "__pycache__"]))


    mode='source_to_source'
    img = 't2f'

    snapshot_path = "./model/{}_{}".format(
    args.model,img)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)
