import torch
import glob
import os
from torchvision import transforms
from PIL import Image, ExifTags
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import random
from scipy import ndimage
import cv2
import GeodisTK
import math
from skimage import morphology
import matplotlib.pyplot as plt


class LinearLesion(torch.utils.data.Dataset):
    def __init__(self, dataset_path, scale, k_fold_test=1, mode='train'):
        super().__init__()
        self.mode = mode
        self.img_path = dataset_path+'/img'
        self.mask_path = dataset_path+'/mask'
        self.image_lists, self.label_lists, self.edge_lists = self.read_list(self.img_path, k_fold_test=k_fold_test)  # 自写函数，读取图片和金标准的列表

        # 定义数据增强方法，每次使用2~4个Augmenter来处理图片,每个batch中的Augmenters顺序不一样
        self.flip = iaa.SomeOf((1, 4), [
             iaa.Fliplr(0.5),  # 水平翻转50%的图像
             iaa.Flipud(0.5),  # 上下翻转
             iaa.Affine(rotate=(-30, 30)),  # 仿射变换：平移、旋转、放缩、错切，rotate: 平移角度
             iaa.AdditiveGaussianNoise(scale=(0.0, 0.08*255))  # 添加高斯噪声
        ], random_order=True)

        # resize，以float64的格式存储的，数值的取值范围是（0~1）
        self.resize_label = transforms.Resize(scale, Image.NEAREST)
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        self.to_tensor = transforms.ToTensor()  # 数据转换

    def __getitem__(self, index):
        # load image and crop
        try:
            self.img = Image.open(self.image_lists[index])

            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(self.img._getexif().items())
            if exif[orientation] == 3:
                self.img = self.img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                self.img = self.img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                self.img = self.img.rotate(90, expand=True)
        except:
            pass
        img = self.img
        img = self.resize_img(img)
        img = np.array(img)

        # --------------------输出结果图-------------------------

        label_path = self.label_lists[index]
        edges = self.edge_lists[index]

        # load label
        if self.mode != 'test':
            label = Image.open(self.label_lists[index])
            label = label.convert('L')  # 灰度图
            label = self.resize_label(label)
            label = np.array(label)

            label[label != 255] = 0
            label[label == 255] = 1

            # *****************************************************
            edge = Image.open(self.edge_lists[index])
            edge = edge.convert('L')  # 灰度图
            edge = self.resize_label(edge)
            edge = np.array(edge)
            #
            # 边缘太细
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            # edge = cv2.dilate(edge, kernel)

            edge[edge != 255] = 0
            edge[edge == 255] = 1

            if self.mode == 'train':
                seq_det = self.flip.to_deterministic()  # 确定一个数据增强的序列
                seg_map = ia.SegmentationMapsOnImage(label, shape=label.shape)  # 方便可视化
                seg_map1 = ia.SegmentationMapsOnImage(edge, shape=edge.shape)

                img = seq_det.augment_image(img)  # 运用在原图
                label = seq_det.augment_segmentation_maps([seg_map])[0].get_arr().astype(np.uint8)  # 运用在金标准
                edge = seq_det.augment_segmentation_maps([seg_map1])[0].get_arr().astype(np.uint8)

            label = np.reshape(label, (1,)+label.shape)  # 增加一维，变成1, 256, 256的数组,(img是三维数组)
            label = torch.from_numpy(label.copy()).float()  # 生成张量
            labels = label

            # ***************************************
            edge = np.reshape(edge, (1,) + edge.shape)
            edge = torch.from_numpy(edge.copy()).float()
            edges = edge

        # img=np.reshape(img,img.shape+(1,))       # 如果输入是1通道需打开此注释 ******
        img = self.to_tensor(img.copy()).float()

        return img, labels, edges  # 训练 验证
        # return img, labels, label_path  # 测试

    def __len__(self):
        return len(self.image_lists)

    def read_list(self, image_path, k_fold_test=1):
        fold = sorted(os.listdir(image_path))  # 返回指定的文件夹包含的文件或文件夹的名字的列表[f1,f2,f3,f4,f5]
        img_list = []
        if self.mode == 'train':
            fold_r = fold
            fold_r.remove('f'+str(k_fold_test))  # remove test_data——f1
            for item in fold_r:  # 遍历序列list
                img_list += glob.glob(os.path.join(image_path, item)+'/*.jpg')  # 路径拼接，获取图片
            label_list = [x.replace('img', 'mask').split('.')[0]+'.png' for x in img_list]  # [0] 输出.前面的内容
            edge_list = [x.replace('img', 'edge').split('.')[0] + '.png' for x in img_list]
        elif self.mode == 'val' or self.mode == 'test':
            fold_s = fold[k_fold_test-1]  # fold[0]--f1(test_data)
            img_list = glob.glob(os.path.join(image_path, fold_s)+'/*.jpg')
            label_list = [x.replace('img', 'mask').split('.')[0]+'.png' for x in img_list]
            edge_list = [x.replace('img', 'edge').split('.')[0] + '.png' for x in img_list]
        return img_list, label_list, edge_list


# 高斯 随机区域提取骨架
class LinearLesion1(torch.utils.data.Dataset):
    def __init__(self, dataset_path, scale, k_fold_test=1, mode='train'):
        super().__init__()
        self.mode = mode
        self.img_path = dataset_path+'/img'
        self.mask_path = dataset_path+'/mask'
        self.image_lists, self.label_lists, self.out_lists, self.edge_lists = self.read_list(self.img_path, k_fold_test=k_fold_test)  # 自写函数，读取图片和金标准的列表

        # 定义数据增强方法，每次使用2~4个Augmenter来处理图片,每个batch中的Augmenters顺序不一样
        self.flip = iaa.SomeOf((1, 4), [
             iaa.Fliplr(0.5),  # 水平翻转50%的图像
             iaa.Flipud(0.5),  # 上下翻转
             iaa.Affine(rotate=(-30, 30)),  # 仿射变换：平移、旋转、放缩、错切，rotate: 平移角度
             iaa.AdditiveGaussianNoise(scale=(0.0, 0.08*255))  # 添加高斯噪声
        ], random_order=True)

        # resize，以float64的格式存储的，数值的取值范围是（0~1）
        self.resize_label = transforms.Resize(scale, Image.NEAREST)
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        self.to_tensor = transforms.ToTensor()  # 数据转换

    def __getitem__(self, index):
        # load image and crop
        try:
            self.img = Image.open(self.image_lists[index])

            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(self.img._getexif().items())
            if exif[orientation] == 3:
                self.img = self.img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                self.img = self.img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                self.img = self.img.rotate(90, expand=True)
        except:
            pass
        img = self.img
        img = self.resize_img(img)
        img = np.array(img)

        # 测地距离
        I = np.asanyarray(img, np.float32)
        S= np.zeros((I.shape[0], I.shape[1]), np.uint8)
        S1 = np.zeros((I.shape[0], I.shape[1]), np.uint8)
        S2 = np.zeros((I.shape[0], I.shape[1]), np.uint8)
        l_x1, l_y1, l_x2, l_y2 = [], [], [], []
        l1, l2 = [], []

        S11 = np.zeros((I.shape[0], I.shape[1]), np.uint8)
        S22 = np.zeros((I.shape[0], I.shape[1]), np.uint8)

        # 连通域像素点个数阈值，判断是否在此连通域取种子点
        pixel_threshold = 30  # 30
        v_key = 30
        v_others = 10

# **********************************************************************************************************************
        if self.mode == 'test':
            # ---------------------------------------------计算性能指标---------------------------------------------------
            paths = self.label_lists[index]
            label = Image.open(self.label_lists[index])
            label = label.convert('L')  # 灰度图
            label = self.resize_label(label)
            label = np.array(label)
            label[label != 255] = 0
            label[label == 255] = 1

            edge = Image.open(self.edge_lists[index])
            edge = edge.convert('L')  # 灰度图
            edge = self.resize_label(edge)
            edge = np.array(edge)
            edge[edge != 255] = 0
            edge[edge == 255] = 1

            out = Image.open(self.out_lists[index])
            out = out.convert('L')  # 灰度图
            out = self.resize_label(out)

            out1 = np.array(out)
            kernel_out = np.ones((5, 5), np.uint8)
            out1 = cv2.morphologyEx(out1, cv2.MORPH_OPEN, kernel_out, iterations=1)
            out1 = cv2.morphologyEx(out1, cv2.MORPH_CLOSE, kernel_out, iterations=1)

            out = np.array(out)
            out[out != 255] = 0
            out[out == 255] = 1

            # 关键点
            num_labels, labels1, stats, centroids = cv2.connectedComponentsWithStats(out1, connectivity=8)
            for i in range(1, len(centroids)):
                S[int(centroids[i][1]), int(centroids[i][0])] = 1

            d = GeodisTK.geodesic2d_raster_scan(I, S, 0.0, 2)
            d = np.exp(-2.772588722 * (d ** 2) / (v_key ** 2))  # 30
            d = np.uint8((cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)) - 255.0)
            d = np.reshape(d, (512, 512, 1))
            d = self.to_tensor(d.copy()).float()

            mix = label - out

            # 少分
            less = np.copy(mix)
            less[less != 1] = 0

            # =============骨架提取================
            skeleton1 = morphology.skeletonize(less)  # 骨架提取要输入'0''1'矩阵
            skeleton1 = skeleton1.astype(np.int)
            skeleton1[skeleton1 == 1] = 255
            skeleton1 = skeleton1.astype(np.uint8)
            # ===================================
            num_labels3, labels3, stats3, centroids3 = cv2.connectedComponentsWithStats(skeleton1, connectivity=8)

            for i in range(1, num_labels3):
                # 骨架内的像素点的(行, 列)坐标
                x11, y11 = np.where(labels3 == i)
                if len(x11) > pixel_threshold:
                    l1.append(i)

            if len(l1) == 0:
                pass
            elif len(l1) == 1:
                x11, y11 = np.where(labels3 == l1[0])
                l_x1.extend(list(x11))
                l_y1.extend(list(y11))
            else:
                p1 = random.randint(1, len(l1))
                # p1 = len(l1)
                l1 = random.sample(l1, p1)
                for i in l1:
                    x11, y11 = np.where(labels3 == i)
                    l_x1.extend(list(x11))
                    l_y1.extend(list(y11))

            for j in range(len(l_x1)):
                S1[l_x1[j], l_y1[j]] = 1

            # 高斯
            d1 = GeodisTK.geodesic2d_raster_scan(I, S1, 0.0, 2)
            d1 = np.exp(-2.772588722 * (d1 ** 2) / (v_others ** 2))
            # d1 = np.uint8(255.0 - (np.minimum(np.maximum(d1, 0.0), 255.0)))
            d1 = np.uint8(cv2.normalize(d1, None, 0, 255, cv2.NORM_MINMAX) - 255.0)
            d1 = np.reshape(d1, (512, 512, 1))

            # 多分
            more = np.copy(mix)
            more[more != 255] = 0
            more[more == 255] = 1

            skeleton2 = morphology.skeletonize(more)  # 骨架提取要输入'0''1'矩阵
            # skeleton2 = morphology.medial_axis(more)  # 中线提取要输入布尔值的矩阵
            skeleton2 = skeleton2.astype(np.int)
            skeleton2 = skeleton2.astype(np.uint8)
            skeleton2[skeleton2 == 1] = 255
            num_labels4, labels4, stats4, centroids4 = cv2.connectedComponentsWithStats(skeleton2, connectivity=8)

            for i in range(1, num_labels4):
                x22, y22 = np.where(labels4 == i)
                if len(x22) > pixel_threshold:
                    l2.append(i)

            if len(l2) == 0:
                pass
            elif len(l2) == 1:
                x22, y22 = np.where(labels4 == l2[0])
                l_x2.extend(list(x22))
                l_y2.extend(list(y22))
            else:
                p2 = random.randint(1, len(l2))
                # p2 = len(l2)
                l2 = random.sample(l2, p2)
                for i in l2:
                    x22, y22 = np.where(labels4 == i)
                    l_x2.extend(list(x22))
                    l_y2.extend(list(y22))

            for n in range(len(l_x2)):
                S2[l_x2[n], l_y2[n]] = 1
            # 测地距离
            d2 = GeodisTK.geodesic2d_raster_scan(I, S2, 0.0, 2)
            d2 = np.exp(-2.772588722 * (d2 ** 2) / (v_others ** 2))
            # d2 = np.uint8(255.0 - (np.minimum(np.maximum(d2, 0.0), 255.0)))
            d2 = np.uint8((cv2.normalize(d2, None, 0, 255, cv2.NORM_MINMAX)) - 255.0)
            d2 = np.reshape(d2, (512, 512, 1))

            # img = np.reshape(img, (512, 512, 1))
            img = np.concatenate((img, d1, d2), axis=2)

            label = np.reshape(label, (1,) + label.shape)
            label = torch.from_numpy(label.copy()).float()
            labels = label

            edge = np.reshape(edge, (1,) + edge.shape)
            edge = torch.from_numpy(edge.copy()).float()
            edges = edge

# **********************************************************************************************************************
        if self.mode == 'val':
            paths = self.label_lists[index]
            label = Image.open(self.label_lists[index])
            label = label.convert('L')  # 灰度图
            label = self.resize_label(label)
            label = np.array(label)
            label[label != 255] = 0
            label[label == 255] = 1

            edge = Image.open(self.edge_lists[index])
            edge = edge.convert('L')  # 灰度图
            edge = self.resize_label(edge)
            edge = np.array(edge)
            edge[edge != 255] = 0
            edge[edge == 255] = 1

            out = Image.open(self.out_lists[index])
            out = out.convert('L')  # 灰度图
            out = self.resize_label(out)

            out1 = np.array(out)
            kernel_out = np.ones((5, 5), np.uint8)
            out1 = cv2.morphologyEx(out1, cv2.MORPH_OPEN, kernel_out, iterations=1)
            out1 = cv2.morphologyEx(out1, cv2.MORPH_CLOSE, kernel_out, iterations=1)

            out = np.array(out)
            out[out != 255] = 0
            out[out == 255] = 1

            # 关键点
            num_labels, labels1, stats, centroids = cv2.connectedComponentsWithStats(out1, connectivity=8)
            for i in range(1, len(centroids)):
                S[int(centroids[i][1]), int(centroids[i][0])] = 1

            d = GeodisTK.geodesic2d_raster_scan(I, S, 0.0, 2)
            d = np.exp(-2.772588722 * (d ** 2) / (v_key ** 2))  # 30
            d = np.uint8((cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)) - 255.0)
            d = np.reshape(d, (512, 512, 1))
            d = self.to_tensor(d.copy()).float()

            mix = label - out

            # 少分
            less = np.copy(mix)
            less[less != 1] = 0

            # =============骨架提取================
            skeleton1 = morphology.skeletonize(less)  # 骨架提取要输入'0''1'矩阵
            skeleton1 = skeleton1.astype(np.int)
            skeleton1[skeleton1 == 1] = 255
            skeleton1 = skeleton1.astype(np.uint8)
            # ===================================
            num_labels3, labels3, stats3, centroids3 = cv2.connectedComponentsWithStats(skeleton1, connectivity=8)

            for i in range(1, num_labels3):
                # 骨架内的像素点的(行, 列)坐标
                x11, y11 = np.where(labels3 == i)
                if len(x11) > pixel_threshold:
                    l1.append(i)

            if len(l1) == 0:
                pass
            elif len(l1) == 1:
                x11, y11 = np.where(labels3 == l1[0])
                l_x1.extend(list(x11))
                l_y1.extend(list(y11))
            else:
                p1 = random.randint(1, len(l1))
                l1 = random.sample(l1, p1)
                for i in l1:
                    x11, y11 = np.where(labels3 == i)
                    l_x1.extend(list(x11))
                    l_y1.extend(list(y11))

            for j in range(len(l_x1)):
                S1[l_x1[j], l_y1[j]] = 1

            # 高斯
            d1 = GeodisTK.geodesic2d_raster_scan(I, S1, 0.0, 2)
            d1 = np.exp(-2.772588722 * (d1 ** 2) / (v_others ** 2))  # 10
            d1 = np.uint8((cv2.normalize(d1, None, 0, 255, cv2.NORM_MINMAX)) - 255.0)
            d1 = np.reshape(d1, (512, 512, 1))

            # 多分
            more = np.copy(mix)
            more[more != 255] = 0
            more[more == 255] = 1

            skeleton2 = morphology.skeletonize(more)  # 骨架提取要输入'0''1'矩阵
            # skeleton2 = morphology.medial_axis(more)  # 中线提取要输入布尔值的矩阵
            skeleton2 = skeleton2.astype(np.int)
            skeleton2[skeleton2 == 1] = 255
            skeleton2 = skeleton2.astype(np.uint8)
            num_labels4, labels4, stats4, centroids4 = cv2.connectedComponentsWithStats(skeleton2, connectivity=8)

            for i in range(1, num_labels4):
                # 骨架内的像素点的(行, 列)坐标
                x22, y22 = np.where(labels4 == i)
                if len(x22) > pixel_threshold:
                    l2.append(i)

            if len(l2) == 0:
                pass
            elif len(l2) == 1:
                x22, y22 = np.where(labels4 == l2[0])
                l_x2.extend(list(x22))
                l_y2.extend(list(y22))
            else:
                p2 = random.randint(1, len(l2))
                l2 = random.sample(l2, p2)
                for i in l2:
                    x22, y22 = np.where(labels4 == i)
                    l_x2.extend(list(x22))
                    l_y2.extend(list(y22))

            for n in range(len(l_x2)):
                S2[l_x2[n], l_y2[n]] = 1

            # 测地距离
            d2 = GeodisTK.geodesic2d_raster_scan(I, S2, 0.0, 2)
            d2 = np.exp(-2.772588722 * (d2 ** 2) / (v_others ** 2))  # 10
            d2 = np.uint8((cv2.normalize(d2, None, 0, 255, cv2.NORM_MINMAX)) - 255.0)
            d2 = np.reshape(d2, (512, 512, 1))

            # out1 = np.reshape(out1, (512, 512, 1))
            # img = np.reshape(img, (512, 512, 1))
            img = np.concatenate((img, d1, d2), axis=2)

            label = np.reshape(label, (1,) + label.shape)  # 增加一维，变成1, 256, 256的数组,(img是三维数组)
            label = torch.from_numpy(label.copy()).float()  # 生成张量
            labels = label

            edge = np.reshape(edge, (1,) + edge.shape)
            edge = torch.from_numpy(edge.copy()).float()
            edges = edge

# **********************************************************************************************************************
        if self.mode == 'train':
            paths = self.label_lists[index]
            label = Image.open(self.label_lists[index])
            label = label.convert('L')  # 灰度图
            label = self.resize_label(label)
            label = np.array(label)
            label[label != 255] = 0
            label[label == 255] = 1

            edge = Image.open(self.edge_lists[index])
            edge = edge.convert('L')  # 灰度图
            edge = self.resize_label(edge)
            edge = np.array(edge)
            edge[edge != 255] = 0
            edge[edge == 255] = 1

            out = Image.open(self.out_lists[index])
            out = out.convert('L')  # 灰度图
            out = self.resize_label(out)

            out1 = np.array(out)
            kernel_out = np.ones((5, 5), np.uint8)
            out1 = cv2.morphologyEx(out1, cv2.MORPH_OPEN, kernel_out, iterations=1)
            out1 = cv2.morphologyEx(out1, cv2.MORPH_CLOSE, kernel_out, iterations=1)
            
            out = np.array(out)
            out[out != 255] = 0
            out[out == 255] = 1

            # 确定一个数据增强的序列
            seq_det = self.flip.to_deterministic()
            # 方便可视化
            seg_map = ia.SegmentationMapsOnImage(label, shape=label.shape)
            seg_out = ia.SegmentationMapsOnImage(out, shape=out.shape)
            seg_out1 = ia.SegmentationMapsOnImage(out1, shape=out1.shape)
            seg_edge = ia.SegmentationMapsOnImage(edge, shape=edge.shape)

            # 运用在原图
            img = seq_det.augment_image(img)
            # 运用在金标准
            label = seq_det.augment_segmentation_maps([seg_map])[0].get_arr().astype(np.uint8)
            out = seq_det.augment_segmentation_maps([seg_out])[0].get_arr().astype(np.uint8)
            out1 = seq_det.augment_segmentation_maps([seg_out1])[0].get_arr().astype(np.uint8)
            edge = seq_det.augment_segmentation_maps([seg_edge])[0].get_arr().astype(np.uint8)

            # 关键点
            num_labels, labels1, stats, centroids = cv2.connectedComponentsWithStats(out1, connectivity=8)
            for i in range(1, len(centroids)):
                S[int(centroids[i][1]), int(centroids[i][0])] = 1

            d = GeodisTK.geodesic2d_raster_scan(I, S, 0.0, 2)
            d = np.exp(-2.772588722 * (d ** 2) / (v_key ** 2))  # 30
            d = np.uint8((cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)) - 255.0)
            d = np.reshape(d, (512, 512, 1))
            d = self.to_tensor(d.copy()).float()

            mix = label - out

            # 少分
            less = np.copy(mix)
            less[less != 1] = 0

            # ********************
            # less[less == 1] = 255
            # less = cv2.morphologyEx(less, cv2.MORPH_OPEN, kernel)
            # cv2.imshow('S2', less)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # **********************

            # =============骨架提取================
            skeleton1 = morphology.skeletonize(less)  # 骨架提取要输入'0''1'矩阵
            skeleton1 = skeleton1.astype(np.int)
            skeleton1[skeleton1 == 1] = 255
            skeleton1 = skeleton1.astype(np.uint8)
            # ===================================
            num_labels3, labels3, stats3, centroids3 = cv2.connectedComponentsWithStats(skeleton1, connectivity=8)

            for i in range(1, num_labels3):
                # 骨架内的像素点的(行, 列)坐标
                x11, y11 = np.where(labels3 == i)
                if len(x11) > pixel_threshold:
                    l1.append(i)

            if len(l1) == 0:
                pass
            elif len(l1) == 1:
                x11, y11 = np.where(labels3 == l1[0])
                l_x1.extend(list(x11))
                l_y1.extend(list(y11))
            else:
                p1 = random.randint(1, len(l1))
                l1 = random.sample(l1, p1)
                for i in l1:
                    x11, y11 = np.where(labels3 == i)
                    l_x1.extend(list(x11))
                    l_y1.extend(list(y11))

            for j in range(len(l_x1)):
                S1[l_x1[j], l_y1[j]] = 1

            # 高斯
            d1 = GeodisTK.geodesic2d_raster_scan(I, S1, 0.0, 2)
            d1 = np.exp(-2.772588722 * (d1 ** 2) / (v_others ** 2))  # 10
            d1 = np.uint8((cv2.normalize(d1, None, 0, 255, cv2.NORM_MINMAX)) - 255.0)
            d1 = np.reshape(d1, (512, 512, 1))

            # 多分
            more = np.copy(mix)
            more[more != 255] = 0

            # ************
            # more = cv2.morphologyEx(more, cv2.MORPH_OPEN, kernel)
            # *****************

            more[more == 255] = 1

            skeleton2 = morphology.skeletonize(more)  # 骨架提取要输入'0''1'矩阵
            # skeleton2 = morphology.medial_axis(more)  # 中线提取要输入布尔值的矩阵
            skeleton2 = skeleton2.astype(np.int)
            skeleton2[skeleton2 == 1] = 255
            skeleton2 = skeleton2.astype(np.uint8)
            num_labels4, labels4, stats4, centroids4 = cv2.connectedComponentsWithStats(skeleton2, connectivity=8)

            for i in range(1, num_labels4):
                # 骨架内的像素点的(行, 列)坐标
                x22, y22 = np.where(labels4 == i)
                if len(x22) > pixel_threshold:
                    l2.append(i)

            if len(l2) == 0:
                pass
            elif len(l2) == 1:
                x22, y22 = np.where(labels4 == l2[0])
                l_x2.extend(list(x22))
                l_y2.extend(list(y22))
            else:
                p2 = random.randint(1, len(l2))
                l2 = random.sample(l2, p2)
                for i in l2:
                    x22, y22 = np.where(labels4 == i)
                    l_x2.extend(list(x22))
                    l_y2.extend(list(y22))

            for n in range(len(l_x2)):
                S2[l_x2[n], l_y2[n]] = 1

            # 测地距离
            d2 = GeodisTK.geodesic2d_raster_scan(I, S2, 0.0, 2)
            d2 = np.exp(-2.772588722 * (d2 ** 2) / (v_others ** 2))  # 10
            d2 = np.uint8((cv2.normalize(d2, None, 0, 255, cv2.NORM_MINMAX)) - 255.0)
            d2 = np.reshape(d2, (512, 512, 1))

            # out1 = np.reshape(out1, (512, 512, 1))
            img = np.concatenate((img, d1, d2), axis=2)

            label = np.reshape(label, (1,) + label.shape)  # 增加一维，变成1, 256, 256的数组,(img是三维数组)
            label = torch.from_numpy(label.copy()).float()  # 生成张量
            labels = label

            edge = np.reshape(edge, (1,) + edge.shape)
            edge = torch.from_numpy(edge.copy()).float()
            edges = edge

        img = self.to_tensor(img.copy()).float()

        return img, labels, edges, d, paths

    def __len__(self):
        return len(self.image_lists)

    def read_list(self, image_path, k_fold_test=1):
        fold = sorted(os.listdir(image_path))  # 返回指定的文件夹包含的文件或文件夹的名字的列表[f1,f2,f3,f4,f5]
        img_list = []
        if self.mode == 'train':
            fold_r = fold
            fold_r.remove('f'+str(k_fold_test))  # remove test_data——f1
            for item in fold_r:  # 遍历序列list
                img_list += glob.glob(os.path.join(image_path, item)+'/*.jpg')  # 路径拼接，获取图片
            label_list = [x.replace('img', 'mask').split('.')[0]+'.png' for x in img_list]  # [0] 输出.前面的内容
            out_list = [x.replace('img', 'out').split('.')[0]+'.png' for x in img_list]
            edge_list = [x.replace('img', 'edge').split('.')[0] + '.png' for x in img_list]
        elif self.mode == 'val' or self.mode == 'test':
            fold_s = fold[k_fold_test-1]  # fold[0]--f1(test_data)
            img_list = glob.glob(os.path.join(image_path, fold_s)+'/*.jpg')
            label_list = [x.replace('img', 'mask').split('.')[0]+'.png' for x in img_list]
            out_list = [x.replace('img', 'out').split('.')[0] + '.png' for x in img_list]
            edge_list = [x.replace('img', 'edge').split('.')[0] + '.png' for x in img_list]
        return img_list, label_list, out_list, edge_list
