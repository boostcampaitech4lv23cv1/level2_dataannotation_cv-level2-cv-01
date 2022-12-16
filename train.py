"""
- latest checkpoint 저장 이름 변경
- best checkpoint 저장 --> val set 이 없어서 train_loss 최소일때 저장.. 이게 맞나?
- seed 고정
- wandb logging
- valid set이 생긴다면 이렇게 하면 될 듯? valid set 생기고 나서 다시 고려해야함. 현재는 주석처리
"""

import os
import os.path as osp
import sys
import time
import math
import json
import wandb
import random
import numpy as np
from pytz import timezone
from datetime import datetime, timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
from detect import get_bboxes
from deteval import calc_deteval_metrics

def seed_everything(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = ArgumentParser()

    # Custom args
    parser.add_argument('--random_seed', type=int, default=2022)
    
    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data'))

    parser.add_argument('--data_format', nargs = '+', type=str, default='ICDAR17_Korean')

    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=1)

    parser.add_argument('--wandb_project', type=str, default='bc_cv01-data_annotation')
    parser.add_argument('--wandb_entity', type=str, default='bc_cv01-data_annotation')
    parser.add_argument('--wandb_run', type=str, default='model')
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(random_seed, data_dir, data_format, model_dir, device, image_size, input_size, num_workers, 
                train_batch_size, valid_batch_size,
                learning_rate, max_epoch, save_interval, wandb_project, wandb_entity, wandb_run):

    seed_everything(random_seed)

    start_time = datetime.now(timezone('Asia/Seoul')).strftime('_%y%m%d_%H%M%S')
    save_dir = osp.join(model_dir, wandb_run + start_time)

    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(osp.join(save_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent = 4)

    # wandb 설정
    wandb.init(
            project=f'{wandb_project}',
            entity=f'{wandb_entity}',
            name = wandb_run + start_time
        )
    wandb.config.update({"run_name": wandb_run,
                         "device": device,
                         "image_size": image_size,
                         "input_size": input_size,
                         "lr": learning_rate, 
                         "channels": 16,
                         "num_workers":num_workers,
                         "train_batch_size":train_batch_size,
                         "valid_batch_size":valid_batch_size,
                         "max_epoch":max_epoch,
                         "model_dir": model_dir,
                         "save_interval":save_interval,
                         "seed": random_seed,
                         "wandbproject":wandb_project,
                         "wandbentity":wandb_entity
                         })
    
    if len(data_format) == 1:
        root_dir = osp.join(data_dir, data_format[0])
        train_dataset = SceneTextDataset(root_dir, split='train_fold0', image_size=image_size, crop_size=input_size)
        valid_dataset = SceneTextDataset(root_dir, split='valid_fold0', image_size=image_size, crop_size=input_size)
    elif len(data_format) == 0:
        raise ValueError
    else:
        train_dataset_list = []
        valid_dataset_list = []
        for format in data_format:
            root_dir = osp.join(data_dir, format)
            train_dataset_list.append(SceneTextDataset(root_dir, split='train', image_size=image_size, crop_size=input_size))
            valid_dataset_list.append(SceneTextDataset(root_dir, split='valid', image_size=image_size, crop_size=input_size))
        train_dataset = ConcatDataset(train_dataset_list)
        valid_dataset = ConcatDataset(valid_dataset_list)
    
    # train_dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
    train_dataset = EASTDataset(train_dataset)
    num_train_batches = math.ceil(len(train_dataset) / train_batch_size)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

    # valid_dataset = SceneTextDataset(data_dir, split='valid', image_size=image_size, crop_size=input_size)
    valid_dataset = EASTDataset(valid_dataset)
    num_valid_batches = math.ceil(len(valid_dataset) / valid_batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    
    model.load_state_dict(torch.load('trained_models/unrealtext_best_loss.pth', 
                                     map_location='cpu'))
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = max_epoch)

    best_loss = 999999999
    best_hmean = -999999999

    with open(osp.join(save_dir, 'log.txt'), 'w') as f:
        for epoch in range(max_epoch):
            model.train()
            epoch_loss, epoch_start = 0, time.time()
            with tqdm(total=num_train_batches) as pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                    pbar.set_description('[Epoch TRAIN {}]'.format(epoch + 1))

                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_train = loss.item()
                    epoch_loss += loss_train

                    pbar.update(1)
                    train_dict = {
                        'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                        'IoU loss': extra_info['iou_loss']
                    }
                    pbar.set_postfix(train_dict)
                    
            scheduler.step()

            wandb.log({"train/Loss": epoch_loss / num_train_batches,
                        "train/Cls_loss": extra_info['cls_loss'],
                        "train/Angle_loss": extra_info['angle_loss'],
                        "train/Iou_loss": extra_info['iou_loss'],
                        })

            f.write("[EPOCH TRAIN {:>03d}] Cls loss={:.4f}, Angle loss={:.4f}, IoU loss={:.4f}, Elapsed time: {}\n".format(
                epoch+1, extra_info['cls_loss'], extra_info['angle_loss'], extra_info['iou_loss'], timedelta(seconds=time.time() - epoch_start)))

            print('TRAIN : Mean loss: {:.4f} | Logged time : {}'.format(
                epoch_loss / num_train_batches, datetime.now(timezone('Asia/Seoul')).strftime('%H:%M:%S')))
            
            f.write('TRAIN : Mean loss: {:.4f} | Logged time : {}'.format(
                epoch_loss / num_train_batches, datetime.now(timezone('Asia/Seoul')).strftime('%H:%M:%S')))

            if (epoch+1) % 3 == 0:
                gt_bboxes = []
                pred_bboxes = []
                trans = []

                with torch.no_grad():
                    model.eval()
                    epoch_loss, epoch_start = 0, time.time()
                    with tqdm(total=num_valid_batches) as pbar:
                        for img, gt_score_map, gt_geo_map, roi_mask in valid_loader:
                            pbar.set_description('[Epoch VALID {}]'.format(epoch + 1))

                            loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                            loss_valid = loss.item()
                            epoch_loss += loss_valid

                            orig_sizes = []
                            gt_bbox = []
                            pred_bbox = []
                            tran = []
                            for i in img:
                                orig_sizes.append(i.shape[1:3])
                            
                            for gt_score, gt_geo, pred_score, pred_geo, orig_size in zip(gt_score_map.numpy(), gt_geo_map.numpy(), extra_info['score_map'].cpu().numpy(), extra_info['geo_map'].cpu().numpy(), orig_sizes):
                                gt_bbox_angle = get_bboxes(gt_score, gt_geo)
                                pred_bbox_angle = get_bboxes(pred_score, pred_geo)
                                if gt_bbox_angle is None:
                                    gt_bbox_angle = np.zeros((0, 4, 2), dtype = np.float32)
                                    tran_angle = []
                                else:
                                    gt_bbox_angle = gt_bbox_angle[:, :8].reshape(-1, 4, 2)
                                    gt_bbox_angle *= max(orig_size) / input_size
                                    tran_angle = ['null' for _ in range(gt_bbox_angle.shape[0])]
                                if pred_bbox_angle is None:
                                    pred_bbox_angle = np.zeros((0, 4, 2), dtype = np.float32)
                                else:
                                    pred_bbox_angle = pred_bbox_angle[:, :8].reshape(-1, 4, 2)
                                    pred_bbox_angle *= max(orig_size) / input_size
                                
                                tran.append(tran_angle)
                                gt_bbox.append(gt_bbox_angle)
                                pred_bbox.append(pred_bbox_angle)
                            
                            gt_bboxes.extend(gt_bbox)
                            pred_bboxes.extend(pred_bbox)
                            trans.extend(tran)
                            
                            pbar.update(1)
                            valid_dict = {
                                'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                                'IoU loss': extra_info['iou_loss']
                            }
                            pbar.set_postfix(valid_dict)
                
                pred_bboxes_dict= dict()
                gt_bboxes_dict = dict() 
                trans_dict = dict()
                for img_num in range(len(valid_dataset)):
                    pred_bboxes_dict[f'img_{img_num}'] = pred_bboxes[img_num]
                    gt_bboxes_dict[f'img_{img_num}'] = gt_bboxes[img_num]
                    trans_dict[f'img_{img_num}'] = trans[img_num]
                deteval_dict = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, trans_dict)
                metric_dict = deteval_dict['total']
                precision = metric_dict['precision']
                recall = metric_dict['recall']
                hmean = metric_dict['hmean']

                wandb.log({"valid/Loss": epoch_loss / num_valid_batches,
                        "valid/Cls_loss": extra_info['cls_loss'],
                        "valid/Angle_loss": extra_info['angle_loss'],
                        "valid/Iou_loss": extra_info['iou_loss'],
                        "valid/precision": precision,
                        "valid/recall": recall,
                        "valid/hmean": hmean,
                        })

                f.write("[EPOCH VALID {:>03d}] Cls loss={:.4f}, Angle loss={:.4f}, IoU loss={:.4f}, Elapsed time: {}\n".format(
                    epoch+1, extra_info['cls_loss'], extra_info['angle_loss'], extra_info['iou_loss'], timedelta(seconds=time.time() - epoch_start)))

                print('VALID : Mean loss: {:.4f} | Precision: {:.5f} | Recall: {:.5f} | Hmean : {:.5f} | Logged time : {}'.format(
                    epoch_loss / num_valid_batches, precision, recall, hmean, datetime.now(timezone('Asia/Seoul')).strftime('%H:%M:%S')))
                
                f.write('VALID : Mean loss: {:.4f} | Precision: {:.5f} | Recall: {:.5f} | Hmean : {:.5f} | Logged time : {}'.format(
                    epoch_loss / num_valid_batches, precision, recall, hmean, datetime.now(timezone('Asia/Seoul')).strftime('%H:%M:%S')))

                if best_hmean < hmean:
                    ckpt_fpath = osp.join(save_dir, 'best_hmean.pth')
                    torch.save(model.state_dict(), ckpt_fpath)
                    best_hmean = hmean

                if best_loss > epoch_loss / num_valid_batches:
                    ckpt_fpath = osp.join(save_dir, 'best_loss.pth')
                    torch.save(model.state_dict(), ckpt_fpath)
                    best_loss = epoch_loss / num_valid_batches

                ckpt_fpath = osp.join(save_dir, f'latest_{epoch+1}epoch.pth')
                torch.save(model.state_dict(), ckpt_fpath)

            ckpt_fpath = osp.join(save_dir, f'latest_every_{epoch + 1}epoch.pth')
            torch.save(model.state_dict(), ckpt_fpath)

def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
