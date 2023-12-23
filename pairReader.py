import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch
import pathlib
import random


def gazeto2d(gaze):
    yaw = np.arctan2(-gaze[0], -gaze[2])
    pitch = np.arcsin(-gaze[1])
    return np.array([yaw, pitch])


class loader(Dataset):
    def __init__(self, path, root, pic_num, header=True, cams=18, pairID=0):
        assert pairID < cams // 2
        self.lines = []
        self.pic_num = pic_num
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    line = f.readlines()
                    if header: line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                self.lines = f.readlines()
                if header: self.lines.pop(0)
        # random.shuffle(self.lines)
        self.idx = []

        length = len(self.lines)
        assert length % cams == 0
        frame_num = length // cams

        sample_num = self.pic_num if self.pic_num >= 0 else frame_num
        frame_idx = random.sample(list(range(frame_num)), sample_num)

        for e in frame_idx:
            # cam_idx = random.sample(list(range(cams)), 2)
            cam_idx = [pairID, pairID + cams // 2]
            # cam_idx = [np.random.randint(0, 10), np.random.randint(0, 10)]
            self.idx.append([e * cams + cam_idx[0], e * cams + cam_idx[1]])
            # self.idx.append([0, 1])
        # print(self.idx)
        self.root = pathlib.Path(root)

    def __len__(self):
        # if self.pic_num < 0:
        return len(self.idx)
        # return self.pic_num

    def __getitem__(self, idx):
        line_idx_1, line_idx_2 = self.idx[idx]
        line_1 = self.lines[line_idx_1]
        line_1 = line_1.strip().split(" ")
        # print(line)

        name_1 = line_1[0].split('/')[0]
        gaze3d_1 = line_1[4]
        head3d_1 = line_1[5]
        face_1 = line_1[0]
        R_mat_1 = line_1[6]

        label_1 = np.array(gaze3d_1.split(",")).astype("float")
        label_1 = torch.from_numpy(label_1).type(torch.FloatTensor)

        headpose_1 = np.array(head3d_1.split(",")).astype("float")
        headpose_1 = torch.from_numpy(headpose_1).type(torch.FloatTensor)

        rmat_1 = np.array(R_mat_1.split(",")).astype("float").reshape(3, 3)
        rmat_1 = torch.from_numpy(rmat_1).type(torch.FloatTensor)

        # print(self.root/name/ face)
        fimg_1 = cv2.imread(str(self.root / face_1))
        fimg_1 = cv2.resize(fimg_1, (448, 448)) / 255.0
        fimg_1 = fimg_1.transpose(2, 0, 1)

        data_1 = {"face": torch.from_numpy(fimg_1).type(torch.FloatTensor),
                  "head_pose": headpose_1, "R_mat": rmat_1,
                  "name": name_1}

        line_2 = self.lines[line_idx_2]
        line_2 = line_2.strip().split(" ")
        # print(line)

        name_2 = line_2[0].split('/')[0]
        gaze3d_2 = line_2[4]
        head3d_2 = line_2[5]
        face_2 = line_2[0]
        R_mat_2 = line_2[6]

        label_2 = np.array(gaze3d_2.split(",")).astype("float")
        label_2 = torch.from_numpy(label_2).type(torch.FloatTensor)

        headpose_2 = np.array(head3d_2.split(",")).astype("float")
        headpose_2 = torch.from_numpy(headpose_2).type(torch.FloatTensor)

        rmat_2 = np.array(R_mat_2.split(",")).astype("float").reshape(3, 3)
        rmat_2 = torch.from_numpy(rmat_2).type(torch.FloatTensor)

        # print(self.root/name/ face)
        fimg_2 = cv2.imread(str(self.root / face_2))
        fimg_2 = cv2.resize(fimg_2, (448, 448)) / 255.0
        fimg_2 = fimg_2.transpose(2, 0, 1)

        data_2 = {"face": torch.from_numpy(fimg_2).type(torch.FloatTensor),
                  "head_pose": headpose_2, "R_mat": rmat_2,
                  "name": name_2}
        # print(rmat_1.T @ rmat_1, rmat_2 @ rmat_2.T)

        return data_1, label_1, data_2, label_2


def txtload(labelpath, imagepath, batch_size, cams=18, pic_num=-1, pairID=0, shuffle=True, num_workers=0, header=True):
    # print(labelpath,imagepath)
    dataset = loader(labelpath, imagepath, pic_num, header, cams, pairID)
    print(f"[Read Data]: Total num: {len(dataset)}")
    # print(f"[Read Data]: Label path: {labelpath}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_everything(18)
    path = ['/home/lrc/gaze_datasets/eth100k/Label_test/subject0008.label',
            '/home/lrc/gaze_datasets/eth100k/Label_test/subject0018.label',
            '/home/lrc/gaze_datasets/eth100k/Label_test/subject0026.label']
    d = txtload(path, '/home/lrc/gaze_datasets/eth100k/Image', batch_size=32, pic_num=5,
                shuffle=False, num_workers=4, header=True)
    print(len(d))
    for i, (data1, label1, data2, label2) in enumerate(d):
        print(i, label1, label2)
        print(data1['face'].shape, data2['face'].shape)
        cv2.imwrite('/home/lrc/aaai22/Dual_Gaze/1.jpg', data1['face'][2].numpy().transpose(1, 2, 0) * 255)
        cv2.imwrite('/home/lrc/aaai22/Dual_Gaze/2.jpg', data2['face'][2].numpy().transpose(1, 2, 0) * 255)
