import numpy as np


def gazeto2d(gaze):
    yaw = np.arctan2(-gaze[0], -gaze[2])
    pitch = np.arcsin(-gaze[1])
    return np.array([yaw, pitch])


def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


def angular(gaze, label):
    gaze = gazeto3d(gaze)
    label = gazeto3d(label)
    total = np.sum(gaze * label)
    return np.arccos(min(total / (np.linalg.norm(gaze) * np.linalg.norm(label)), 0.9999999)) * 180 / np.pi


def angular3d(gaze, label):
    total = np.sum(gaze * label)
    return np.arccos(min(total / (np.linalg.norm(gaze) * np.linalg.norm(label)), 0.9999999)) * 180 / np.pi


def angle2matrix(angles):
    x, y, z = angles[0], angles[1], angles[2]
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R


def matrix2angle(m):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY, the object will be rotated with the order of [rx, ry, rz]
    """
    x = np.arctan2(m[2, 1], m[2, 2])
    y = np.arcsin(-m[2, 0])
    z = np.arctan2(m[1, 0], m[0, 0])
    return x, y, z


def calc_metric3d(path, cams):  # calculate metrics among all pairs
    fo = open(path)
    lines = fo.readlines()
    lines.pop(0)
    lines.pop(-1)
    mono_err = []
    dual_s_err = []
    dual_a_err = []
    hp_err = []
    n = len(lines)
    for i in range(n):
        line = lines[i].split()
        gaze3d = np.array(line[1].split(',')).astype(np.float64)
        gazegt = np.array(line[2].split(',')).astype(np.float64)
        hp = np.array(line[3].split(',')).astype(np.float64)
        hpgt = np.array(line[4].split(',')).astype(np.float64)
        R = angle2matrix(hp)
        theta = np.rad2deg(np.arccos(angle2matrix(hpgt)[2, 2]))

        err = angular3d(gaze3d, gazegt)
        mono_err.append(err)
        hpe = np.mean(np.abs(hpgt - hp))
        hp_err.append(hpe)

        if i % cams < cams // 2:
            dual_s_err.append(err)
            pair = lines[i + cams // 2].split()
        else:
            pair = lines[i - cams // 2].split()

        pairgaze = np.array(pair[1].split(',')).astype(np.float64)
        pairgt = np.array(pair[2].split(',')).astype(np.float64)
        pairhp = np.array(pair[3].split(',')).astype(np.float64)
        pairhpgt = np.array(pair[4].split(',')).astype(np.float64)
        pairR = angle2matrix(pairhp)

        pairtheta = np.rad2deg(np.arccos(angle2matrix(pairhpgt)[2, 2]))
        if pairtheta < theta:
            dual_s_err.append(angular3d(pairgaze, pairgt))
        else:
            dual_s_err.append(err)

        pair_gaze = R @ pairR.T @ pairgaze
        avg_gaze = pair_gaze + gaze3d
        avg_gaze /= np.linalg.norm(avg_gaze)

        dual_a_err.append(angular3d(avg_gaze, gazegt))

    print(f'Mono err:{np.mean(mono_err)}; Dual-S err:{np.mean(dual_s_err)}; Dual-A err:{np.mean(dual_a_err)}')
    return np.mean(mono_err), np.mean(dual_s_err), np.mean(dual_a_err), np.rad2deg(np.mean(hp_err))


def calc_metric_2cam3d(path, cams, pairID):  # calculate metrics for a specific pair
    fo = open(path)
    lines = fo.readlines()
    lines.pop(0)
    lines.pop(-1)
    mono_err, dual_s_err, dual_a_err, hp_err = [], [], [], []
    n = len(lines)
    for i in range(n):
        if not i % cams in [pairID, pairID + cams // 2]:
            continue
        line = lines[i].split()
        gaze3d = np.array(line[1].split(',')).astype(np.float64)
        gazegt = np.array(line[2].split(',')).astype(np.float64)
        hp = np.array(line[3].split(',')).astype(np.float64)
        hpgt = np.array(line[4].split(',')).astype(np.float64)
        R = angle2matrix(hp)
        theta = np.rad2deg(np.arccos(angle2matrix(hpgt)[2, 2]))

        err = angular3d(gaze3d, gazegt)
        mono_err.append(err)
        hpe = np.mean(np.abs(hpgt - hp))
        hp_err.append(hpe)

        if i % cams == pairID:
            # dual_s_err.append(err)
            pair = lines[i + cams // 2].split()
        else:
            pair = lines[i - cams // 2].split()

        pairgaze = np.array(pair[1].split(',')).astype(np.float64)
        pairgt = np.array(pair[2].split(',')).astype(np.float64)
        pairhp = np.array(pair[3].split(',')).astype(np.float64)
        pairhpgt = np.array(pair[4].split(',')).astype(np.float64)
        pairR = angle2matrix(pairhp)

        pairtheta = np.rad2deg(np.arccos(angle2matrix(pairhpgt)[2, 2]))
        if pairtheta < theta:
            dual_s_err.append(angular3d(pairgaze, pairgt))
        else:
            dual_s_err.append(err)

        pair_gaze = R @ pairR.T @ pairgaze
        avg_gaze = pair_gaze + gaze3d
        avg_gaze /= np.linalg.norm(avg_gaze)

        dual_a_err.append(angular3d(avg_gaze, gazegt))
    print(f'Mono err:{np.mean(mono_err):.2f}; Dual-S err:{np.mean(dual_s_err):.2f};'
          f' Dual-A err:{np.mean(dual_a_err):.2f}; HP err:{np.rad2deg(np.mean(hp_err)):.2f}')
    return np.mean(mono_err), np.mean(dual_s_err), np.mean(dual_a_err), np.rad2deg(np.mean(hp_err))


if __name__ == "__main__":
    pid = 7
    baselog = f'/home/lrc/cvpr23/Gaze3d/tune/gaze360_eth2/evaluation/cross-eth100k/20.log'
    # baselog = f'/home/lrc/cvpr23/Gaze3d/eth100k_exp/evaluation/cross-eth100k/10.log'
    print('------\nbase pair', end=': ')
    calc_metric_2cam3d(baselog, cams=18, pairID=pid)
    for i in range(1, 11, 1):
        path = f'/home/lrc/iccv23/ref_Gaze/ablation/gaze3602eth/cnn_mut_stb_sg/evaluation/pair{pid}/cross-eth100k/{i}.log'  # ours
        print(i, end=': ')
        # calc_metric3d(path, cams=18)
        calc_metric_2cam3d(path, cams=18, pairID=pid)
