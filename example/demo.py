import numpy as np
import poselib


def build_random_matrix(num=5):
    m = np.random.rand(num * 3, 3)
    return m


def find_Rt(pts1, pts2, H):
    if np.linalg.matrix_rank(H) < 3:
        raise ValueError(
            "rank of H = {}, expecting 3".format(np.linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < 0, reflection detected!, correcting for it ...\n");
        Vt[1, :] *= -1
        R = np.dot(Vt.T, U.T)
    R3 = np.identity(3)
    R3[0, 0] = R[0, 0]
    R3[2, 2] = R[1, 1]
    R3[0, 2] = R[0, 1]
    R3[2, 0] = R[1, 0]

    centroid_A = np.mean(pts1, axis=0)
    centroid_B = np.mean(pts2, axis=0)
    t = -np.dot(R3, centroid_A) + centroid_B
    return R3, t


pts1 = build_random_matrix()
pts2 = build_random_matrix()
res = poselib.essential_matrix_5pt(pts1, pts2)
for e in res:
    print("shape: {}".format(e.shape))
    if np.linalg.matrix_rank(e) < 3:
        continue

    R, t = find_Rt(pts1, pts2, e)
    print("===R: {}, t: {} ====".format(R, t))
