import os
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import spectral
import matplotlib.pyplot as plt

class ImgRegistration:
    def __init__(self, msi_board_path, RGB_board_path):
        self.msi_board_path = msi_board_path
        self.RGB_board_path = RGB_board_path
        self.M = self.get_rectify_M()
        # np.save('/Users/gordon/code/python/guangxin/coffee/reg3', self.M)
        # np.save(r'D:\code\咖啡豆线扫系统对接\Multimodal-camera\coffee.npy', self.M)

    def read_pngs(self, path):
        """
        传入png图片文件夹路径，读取所有png图片形成numpy二维矩阵
        :param path:
        :return:
        """
        datanames = os.listdir(path)
        datanames = sorted(datanames)
        img_path = path + '/' + str(datanames[0])
        img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), 0)
        a, b = img.shape
        c = len(datanames)

        msi_arr = np.zeros([a, b, c], dtype=np.uint8)
        for i in range(len(datanames)):
            if datanames[i][-3:] == 'png':
                img_path = path + '/' + str(datanames[i])
                img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), 0)
                msi_arr[:, :, i] = img
        return msi_arr

    def get_rectify_M(self):
        """
        获取图像配准变换矩阵M
        1.显示特征匹配图片  2.显示模板图片对正效果图
        """
        msi = spectral.open_image(self.msi_board_path).load()
        msi =  cv2.flip(msi, 1)

        # 归一化到 0-1
        msi_normalized = msi[:, :, int(msi.shape[2]/2)]
        msi_normalized = (msi_normalized - msi_normalized.min()) / (msi_normalized.max() - msi_normalized.min())

        # 限制最小值和最大值到 0 和 255
        msi_clipped = np.clip(msi_normalized, 0, 1)

        # 转换为无符号 8 位整数
        msi_8bit = (msi_clipped * 255).astype(np.uint8)

        spe_gray = msi_8bit

        ### 显示图像
        plt.imshow(spe_gray, cmap='gray')


        RGB_img = cv.imdecode(np.fromfile(self.RGB_board_path, dtype=np.uint8), 0)

        rgb_gray = RGB_img
        sift = cv.SIFT_create()
        # 利用SIFT算法寻找特征点
        kp1, des1 = sift.detectAndCompute(spe_gray, None)
        kp2, des2 = sift.detectAndCompute(rgb_gray, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]

        good = []
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.9 * n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
                matchesMask[i] = [1, 0]

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)
        img3 = cv.drawMatchesKnn(spe_gray, kp1, rgb_gray, kp2, matches, None, **draw_params)
        plt.title('feature matching')
        plt.imshow(img3)
        # plt.show()
        rows, cols = spe_gray.shape[:2]
        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        else:
            raise Exception("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        return M

    def rectify_msi(self, msi_path):
        global rgb_img
        rgb_img = cv.imdecode(np.fromfile(self.RGB_board_path, dtype=np.uint8), 0)

        rows, cols = rgb_img.shape[:2]

        msi = spectral.open_image(msi_path).load()
        msi_imgs = cv2.flip(msi, 1)

        warpImgs = cv.warpPerspective(msi_imgs, np.linalg.inv(self.M), (cols, rows), flags=cv.WARP_INVERSE_MAP)

        return warpImgs



msi_board_path = rf'H:\1\newdata20250806_112536.hdr'
rgb_board_path = rf'H:\1\1.bmp'
reg = ImgRegistration(msi_board_path, rgb_board_path)

msi_path = rf'H:\1\newdata20250806_112536.hdr'
res = reg.rectify_msi(msi_path)
plt.show()