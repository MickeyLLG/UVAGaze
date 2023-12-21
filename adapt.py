import model
import argparse
import pairReader as reader_target
import reader_random as reader_source
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import copy
import yaml
import torch.nn.functional as F
import random
import cv2


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def angle2matrix(angles):  # euler angle to rotation matrix
    x, y, z = angles[0], angles[1], angles[2]
    # x
    Rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(x), -torch.sin(x)],
                       [0, torch.sin(x), torch.cos(x)]], requires_grad=True)
    # y
    Ry = torch.tensor([[torch.cos(y), 0, torch.sin(y)],
                       [0, 1, 0],
                       [-torch.sin(y), 0, torch.cos(y)]], requires_grad=True)
    # z
    Rz = torch.tensor([[torch.cos(z), -torch.sin(z), 0],
                       [torch.sin(z), torch.cos(z), 0],
                       [0, 0, 1]], requires_grad=True)
    R = Rz @ Ry @ Rx
    return R


def matrix2angle(m):  # rotation matrix to euler angle
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY, the object will be rotated with the order of [rx, ry, rz]
    """
    x = torch.arctan2(m[2, 1], m[2, 2])
    y = torch.arcsin(-m[2, 0])
    z = torch.arctan2(m[1, 0], m[0, 0])
    return torch.tensor([x, y, z], requires_grad=True)


def bangle2matrix(angles, device):  # euler angle to rotation matrix in a batch
    Rots = torch.Tensor().to(device)
    for i in range(angles.shape[0]):
        R = angle2matrix(angles[i]).to(device)
        Rots = torch.cat((Rots, R.reshape(1, 3, 3)), 0)
    return Rots


def bmatrix2angle(matrix, device):  # rotation matrix to euler angle in a batch
    angles = torch.Tensor().to(device)
    for i in range(matrix.shape[0]):
        theta = matrix2angle(matrix[i, :, :]).to(device)
        angles = torch.cat((angles, theta.reshape(1, 3)), 0)
    return angles


if __name__ == "__main__":
    seed_everything(1)
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--i', type=int, default=-1,
                        help="i represents the i-th folder used as the test set")
    parser.add_argument('--lr', type=float, default=0.00001, help="learning rate")
    parser.add_argument('--savepath', type=str, default=None, help="path to save models")
    parser.add_argument('--pic', type=int, default=128, help="number of target images")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    parser.add_argument('--cams', type=int, default=18, help="number of cameras on the target dataset")
    parser.add_argument('--gpu', type=int, default=0, help="gpu")
    parser.add_argument('--source', type=str, default='gaze360', help="source dataset, gaze360 or eth-mv(100k)-train")
    parser.add_argument('--target', type=str, default='eth-mv', help="target dataset, eth-mv(100k)")
    parser.add_argument('--stb', action='store_true', help="whether to use stb loss")
    parser.add_argument('--pre', action='store_true', help="whether to use pre loss")
    parser.add_argument('--name', type=str, default='UVAGaze', help="model name")
    parser.add_argument('--pairID', type=int, default=0, help="which camera pair to adapt to")
    args = parser.parse_args()

    config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    config = config["dual"]

    loc = f'cuda:{args.gpu}'

    imagepath_target = config[args.target]["image"]
    labelpath_target = config[args.target]["label"]
    imagepath_source = config[args.source]["image"]
    labelpath_source = config[args.source]["label"]
    modelname = args.name
    bs = args.bs

    folder_target = os.listdir(labelpath_target)
    folder_target.sort()
    folder_source = os.listdir(labelpath_source)
    folder_source.sort()

    # i represents the i-th folder used as the test set. Usually set as -1 to use all folders.
    i = args.i

    if i in list(range(len(folder_target))):
        tests_target = copy.deepcopy(folder_target)
        trains_target = [tests_target.pop(i)]
        print(f"Train Set:{trains_target}")
    else:
        trains_target = copy.deepcopy(folder_target)
        print(f"Train Set:{trains_target}")

    trainlabelpath_target = [os.path.join(labelpath_target, j) for j in trains_target]
    trainlabelpath_source = [os.path.join(labelpath_source, j) for j in folder_source]

    savepath = os.path.join(args.savepath, f"checkpoint/pair{args.pairID}")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    device = torch.device(loc if torch.cuda.is_available() else "cpu")

    print("Read data")
    dataset_source = reader_source.txtload(trainlabelpath_source, imagepath_source, bs,
                                           shuffle=True, num_workers=4, pic_num=-1, header=True)
    dataset_target = reader_target.txtload(trainlabelpath_target, imagepath_target, bs, cams=args.cams,
                                           pairID=args.pairID,
                                           shuffle=False, num_workers=4, pic_num=args.pic, header=True)

    print("Model building")
    pre_model = config[f'pretrain_{args.source}']
    print('pre-model:', pre_model)
    net = model.GazeStatic()
    net.to(device)

    statedict_net = torch.load(pre_model, map_location=loc)
    net.load_state_dict(statedict_net)
    net.eval()

    print("optimizer building")

    gaze_loss_op = model.AngularLoss()
    hp_loss_op = nn.L1Loss().cuda()
    stb_term_op = model.StableLossTerm()
    base_lr = args.lr

    decaysteps = config["params"]["decay_step"]
    decayratio = config["params"]["decay"]

    optimizer = optim.Adam(net.parameters(), lr=base_lr, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

    print("Traning")
    length_target = len(dataset_target)
    total = length_target * config["params"]["epoch"]
    cur = 0
    timebegin = time.time()
    cam_relative = None

    with open(os.path.join(savepath, "train_log"), 'w') as outfile, \
            open(os.path.join(savepath, "loss_log"), 'w') as lossfile:
        for epoch in range(1, config["params"]["epoch"] + 1):
            for i, (target, source) in enumerate(zip(dataset_target, dataset_source)):
                # Acquire data
                data_source, label_source = source
                data1, _, data2, _ = target
                face1 = data1['face'].to(device)
                face2 = data2['face'].to(device)
                face_source = data_source['face'].to(device)
                label_source = label_source.to(device)

                # update rec
                gaze1, hp1 = net(face1)
                gaze2, hp2 = net(face2)
                gaze_source, _ = net(face_source)

                Rots1 = bangle2matrix(hp1, device)
                Rots2 = bangle2matrix(hp2, device)

                gaze1_hcs = torch.bmm(Rots1.permute(0, 2, 1).detach(), gaze1.reshape(bs, 3, 1)).reshape(bs, 3)
                gaze2_hcs = torch.bmm(Rots2.permute(0, 2, 1).detach(), gaze2.reshape(bs, 3, 1)).reshape(bs, 3)
                infront = abs(Rots1[:, 2, 2]) > abs(Rots2[:, 2, 2])
                infront = infront.reshape(bs, 1).repeat(1, 3)
                pseudo = torch.where(infront, gaze1_hcs, gaze2_hcs).detach()

                cam_rots = stb_term_op(hp1, hp2)

                # loss calculation
                if cam_relative is None:
                    cam_relative = cam_rots.detach()

                hp_diff = torch.abs(cam_rots - cam_relative)
                hp_loss = torch.mean(hp_diff)

                mut_loss = gaze_loss_op(gaze1_hcs, pseudo) + gaze_loss_op(gaze2_hcs, pseudo)
                source_loss = gaze_loss_op(gaze_source, label_source)

                loss = 1 * mut_loss
                if args.stb:
                    loss += 50 * hp_loss
                if args.pre:
                    loss += 10 * source_loss

                # update model and cam_relative
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()

                cam_relative = cam_relative * 0.99 + 0.01 * torch.mean(cam_rots, dim=0).detach()

                cur += 1

                # print logs
                if i == 0:
                    timeend = time.time()
                    resttime = (timeend - timebegin) / cur * (total - cur) / 3600
                    log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length_target}] " \
                          f"bs: {bs} pic:{args.pic} stb:{args.stb} pre:{args.pre} stb_loss:{hp_loss:.4f} " \
                          f"pre_loss:{source_loss:.4f} mut_loss:{mut_loss:.4f} loss:{loss:.4f} " \
                          f"lr:{optimizer.state_dict()['param_groups'][0]['lr'] * 10 ** 5:.1f}e-5 " \
                          f", rest time:{resttime:.2f}h"
                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()
                    outfile.flush()
                    lossfile.flush()
            scheduler.step()
            lossfile.flush()
            if epoch % config["save"]["step"] == 0:
                torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))
