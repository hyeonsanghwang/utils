import numpy as np
import cv2
from sklearn.decomposition import PCA


def get_pca(data, n):
    pca = PCA(n_components=n).fit_transform(data)
    return pca


def get_pca_frame(pca,
                  size=1000,
                  cluster=None,
                  selected_idx=None,
                  colors=((255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 128, 128), (0, 0, 0)),
                  selected_color=(165, 123, 41),
                  bi_norm=False):
    px, py = pca[:, 0], pca[:, 1]

    if bi_norm:
        pxmax, pxmin = px.max(), px.min()
        pymax, pymin = py.max(), py.min()

        nx = (((px - pxmin) / (pxmax - pxmin)) * size).astype(np.int32)
        ny = (((py - pymin) / (pymax - pymin)) * size).astype(np.int32)
    else:
        pmax, pmin = pca.max(), pca.min()

        nx = (((px - pmin) / (pmax - pmin)) * size).astype(np.int32)
        ny = (((py - pmin) / (pmax - pmin)) * size).astype(np.int32)

    frame = np.ones((size, size, 3))

    target_xs = nx[:-1]
    target_ys = ny[:-1]

    circle_size = 10
    border_size = 5
    if cluster is None:
        for i, (x, y) in enumerate(zip(target_xs, target_ys)):
            if i == selected_idx:
                cv2.circle(frame, (x, y), circle_size+border_size, (0,0,0), -1)
                cv2.circle(frame, (x, y), circle_size, colors[1], -1)
            else:
                cv2.circle(frame, (x, y), circle_size + border_size, (0, 0, 0), -1)
                cv2.circle(frame, (x, y), circle_size, colors[0], -1)
    else:
        for i, n in enumerate(range(cluster.min(), cluster.max() + 1)):
            for j, (x, y) in enumerate(zip(target_xs[cluster == n], target_ys[cluster == n])):
                if j == selected_idx:
                    cv2.circle(frame, (x, y), circle_size + border_size, (0, 0, 0), -1)
                    cv2.circle(frame, (x, y), circle_size, selected_color, -1)
                else:
                    cv2.circle(frame, (x, y), circle_size + border_size, (0, 0, 0), -1)
                    cv2.circle(frame, (x, y), circle_size, colors[i], -1)

    cv2.circle(frame, (nx[-1], ny[-1]), circle_size, (0, 0, 0), -1)
    return frame
