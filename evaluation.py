import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

import matplotlib.pyplot as plt

from detect import detect
import numpy as np
from deteval import calc_deteval_metrics


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default='/opt/ml/input/data/upstage')
    parser.add_argument('--gt_fname', default='upstage_rare_valid.json')
    parser.add_argument('--model_fname', default='ICDAR17ko_en_upstage_aihub_aug_lr3e4_512_256_BAEK_221214_043327/best_loss.pth')
    parser.add_argument('--output_dir', default='/opt/ml/level2_dataannotation_cv-level2-cv-01/evaluations')

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=20)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_evaluation(model, ckpt_fpath, data_dir, gt_fname, input_size, batch_size):
    model.load_state_dict(torch.load(ckpt_fpath))
    model.eval()

    by_sample_bboxes = []
    images = []
    image_fnames = list(json.load(open(os.path.join(data_dir,'ufo',gt_fname)))['images'].keys())

    for image_fname in tqdm(image_fnames[:]):
        image_fpath = os.path.join(data_dir,'images',image_fname)
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    return image_fnames, by_sample_bboxes


def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get paths to checkpoint files
    ckpt_fpath = os.path.join('/opt/ml/level2_dataannotation_cv-level2-cv-01/trained_models',args.model_fname)

    if not osp.exists(os.path.join(args.output_dir,args.model_fname.split('/')[0])):
        os.makedirs(os.path.join(args.output_dir,args.model_fname.split('/')[0]))

    print('Evaluation in progress')
    image_fname, by_sample_bboxes = do_evaluation(model, ckpt_fpath, args.data_dir, args.gt_fname, args.input_size, args.batch_size)
    
    
    gt = json.load(open(os.path.join(args.data_dir,'ufo',args.gt_fname)))

    by_sample_gt = []
    trans = []
    for fname in image_fname:
        temp = []
        temp2 = []
        for i in list(gt['images'][fname]['words'].keys()):
            temp.append(gt['images'][fname]['words'][str(i)]['points'])
            temp2.append(gt['images'][fname]['words'][str(i)]['transcription'])
        by_sample_gt.append(temp)
        trans.append(temp2)
    by_sample_gt = np.array(by_sample_gt)


    hmeans = []
    for img_num in range(len(by_sample_bboxes)):
        pred_bboxes_dict = dict()
        gt_bboxes_dict = dict()
        trans_dict = dict()

        pred_bboxes_dict[fname] = by_sample_bboxes[img_num]
        gt_bboxes_dict[fname] = by_sample_gt[img_num]
        trans_dict[fname] = trans[img_num]

        deteval_dict = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, trans_dict)['total']
        hmean = deteval_dict['hmean']

        hmeans.append(hmean)
    

    result = []
    for i in range(len(hmeans)):
        result.append((image_fname[i],hmeans[i],by_sample_bboxes[i],by_sample_gt[i]))
    
    result.sort(key=lambda x:x[1])

    
    fig, axes = plt.subplots(4,4,figsize=(30,30))

    for num in range(5):
        for i, (image_fname, hmean, pred_bbox, gt_bbox) in enumerate(result[:12]+result[-4:]):
            img = cv2.imread(os.path.join('/opt/ml/input/data',args.data_dir,'images',f'{image_fname}'))
            for bbox in pred_bbox:
                points = np.array([[int(p[0]),int(p[1])] for p in bbox])

                img = cv2.polylines(img, [points], True, (0,0,225), 3)
                cv2.putText(img,f"pred:{hmean} | 0",(points[0][0],points[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                cv2.putText(img,"1",(points[1][0],points[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                cv2.putText(img,"2",(points[2][0],points[2][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                cv2.putText(img,"3",(points[3][0],points[3][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)

            for bbox in gt_bbox:
                points = np.array([[int(p[0]),int(p[1])] for p in bbox])

                img = cv2.polylines(img, [points], True, (0,255,225), 3)
                cv2.putText(img,"gt | 0",(points[0][0],points[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 2)
                cv2.putText(img,"1",(points[1][0],points[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 2)
                cv2.putText(img,"2",(points[2][0],points[2][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 2)
                cv2.putText(img,"3",(points[3][0],points[3][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 2)

            axes[i//4,i%4].set_title(f'{image_fname} | {hmean}')
            axes[i//4,i%4].plot()
            axes[i//4,i%4].imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            axes[i//4,i%4].axis('off')

        fig.tight_layout()
        plt.savefig(os.path.join(args.output_dir,args.model_fname.split('/')[0],f'{args.gt_fname}_viz_{num}.jpg'))
        result = result[12:-4]


if __name__ == '__main__':
    args = parse_args()
    main(args)
