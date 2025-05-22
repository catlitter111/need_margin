#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步多线程瓶子检测模块
使用多线程异步操作提高NPU使用率，显著提升检测帧率
"""

import cv2
import numpy as np
import logging
from queue import Queue
from rknn.api import RKNN
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor
import threading

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("bottle_detector_async")

# YOLO检测参数
OBJ_THRESH = 0.2
NMS_THRESH = 0.5

# COCO类别
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

# 颜色调色板
color_palette = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def init_rknn_lite(model_path, core_id=0):
    """初始化单个RKNN Lite实例"""
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        logger.error(f"加载RKNN模型失败: {model_path}")
        return None
        
    # 根据core_id分配到不同的NPU核心
    if core_id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif core_id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif core_id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    else:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        
    if ret != 0:
        logger.error(f"初始化运行时环境失败，核心ID: {core_id}")
        return None
        
    logger.info(f"成功初始化RKNN实例，核心ID: {core_id}")
    return rknn_lite


class AsyncBottleDetector:
    """异步多线程瓶子检测器类"""
    
    def __init__(self, model_path, model_size=(640, 640), num_threads=3):
        """
        初始化异步瓶子检测器
        
        参数:
        model_path -- RKNN模型路径
        model_size -- 模型输入尺寸
        num_threads -- 线程数量（建议1-3，对应NPU核心数）
        """
        self.model_path = model_path
        self.model_size = model_size
        self.num_threads = min(num_threads, 3)  # 最多3个核心
        
        # 初始化RKNN实例池
        self.rknn_pool = []
        self.init_pool()
        
        # 创建线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)
        
        # 任务队列
        self.task_queue = Queue()
        self.result_queue = Queue(maxsize=10)  # 限制结果队列大小
        
        # 任务计数器
        self.task_counter = 0
        self.task_lock = threading.Lock()
        
        # 运行标志
        self.running = True
        
    def init_pool(self):
        """初始化RKNN实例池"""
        for i in range(self.num_threads):
            rknn_instance = init_rknn_lite(self.model_path, i % 3)
            if rknn_instance:
                self.rknn_pool.append(rknn_instance)
            else:
                logger.error(f"初始化第{i}个RKNN实例失败")
                
        if not self.rknn_pool:
            raise RuntimeError("无法初始化任何RKNN实例")
            
        logger.info(f"成功初始化{len(self.rknn_pool)}个RKNN实例")
    
    def detect_async(self, image):
        """
        异步检测图像中的瓶子
        
        参数:
        image -- 输入图像
        """
        if not self.running:
            return
            
        with self.task_lock:
            task_id = self.task_counter
            self.task_counter += 1
            
        # 选择RKNN实例
        rknn_instance = self.rknn_pool[task_id % len(self.rknn_pool)]
        
        # 提交检测任务
        future = self.thread_pool.submit(self._detect_worker, rknn_instance, image, task_id)
        self.task_queue.put((task_id, future))
    
    def _detect_worker(self, rknn_instance, image, task_id):
        """检测工作线程"""
        try:
            # 预处理图像
            img = self._letter_box(image, self.model_size)
            input_tensor = np.expand_dims(img, axis=0)
            
            # 执行推理
            outputs = rknn_instance.inference([input_tensor])
            
            # 后处理
            boxes, classes, scores = self._post_process(outputs)
            
            # 提取瓶子检测结果
            bottle_detections = []
            if boxes is not None:
                img_h, img_w = image.shape[:2]
                x_factor = img_w / self.model_size[0]
                y_factor = img_h / self.model_size[1]
                
                for box, score, cl in zip(boxes, scores, classes):
                    if CLASSES[cl] == 'bottle':
                        x1, y1, x2, y2 = [int(_b) for _b in box]
                        
                        left = int(x1 * x_factor)
                        top = int(y1 * y_factor)
                        right = int(x2 * x_factor)
                        bottom = int(y2 * y_factor)
                        
                        center_x = (left + right) // 2
                        center_y = (top + bottom) // 2
                        
                        bottle_detections.append((left, top, right, bottom, score, center_x, center_y))
            
            return task_id, bottle_detections, image
            
        except Exception as e:
            logger.error(f"检测任务{task_id}失败: {e}")
            return task_id, [], image
    
    def get_result(self, timeout=0.1):
        """
        获取检测结果（非阻塞）
        
        返回:
        (detections, image) 或 (None, None)如果没有结果
        """
        # 处理已完成的任务
        while not self.task_queue.empty():
            try:
                task_id, future = self.task_queue.get_nowait()
                if future.done():
                    result_id, detections, image = future.result()
                    # 将结果放入结果队列
                    if not self.result_queue.full():
                        self.result_queue.put((detections, image))
            except:
                pass
        
        # 从结果队列获取最新结果
        try:
            if not self.result_queue.empty():
                return self.result_queue.get_nowait()
        except:
            pass
            
        return None, None
    
    def release(self):
        """释放资源"""
        self.running = False
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        
        # 释放RKNN实例
        for rknn in self.rknn_pool:
            rknn.release()
            
        logger.info("异步检测器资源已释放")
    
    def draw_detection(self, image, detection, distance=None):
        """在图像上绘制瓶子检测结果"""
        left, top, right, bottom, score = detection[:5]
        bottle_class_id = CLASSES.index('bottle')
        color = color_palette[bottle_class_id]
        
        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, 2)
        
        if distance is not None:
            label = f"bottle: {score:.2f}, distance: {distance:.2f}m"
        else:
            label = f"bottle: {score:.2f}"
            
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = left
        label_y = top - 10 if top - 10 > label_height else top + 10
        cv2.rectangle(image, (int(label_x), int(label_y - label_height)), 
                     (int(label_x + label_width), int(label_y + label_height)), color, cv2.FILLED)
        cv2.putText(image, label, (int(label_x), int(label_y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # ========== 以下是原有的辅助函数 ==========
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _letter_box(self, im, new_shape, pad_color=(255, 255, 255), info_need=False):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
        if info_need is True:
            return im, ratio, (dw, dh)
        else:
            return im
    
    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
        scores = (class_max_score * box_confidences)[_class_pos]
        boxes = boxes[_class_pos]
        classes = classes[_class_pos]
        return boxes, classes, scores
    
    def _nms_boxes(self, boxes, scores):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        areas = w * h
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= NMS_THRESH)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep
    
    def _softmax(self, x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)
    
    def _dfl(self, position):
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        y = position.reshape(n, p_num, mc, h, w)
        y = self._softmax(y, 2)
        acc_metrix = np.array(range(mc), dtype=float).reshape(1, 1, mc, 1, 1)
        y = (y * acc_metrix).sum(2)
        return y
    
    def _box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.model_size[1] // grid_h, self.model_size[0] // grid_w]).reshape(1, 2, 1, 1)
        position = self._dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
        return xyxy
    
    def _post_process(self, input_data):
        boxes, scores, classes_conf = [], [], []
        defualt_branch = 3
        pair_per_branch = len(input_data) // defualt_branch
        for i in range(defualt_branch):
            boxes.append(self._box_process(input_data[pair_per_branch * i]))
            classes_conf.append(input_data[pair_per_branch * i + 1])
            scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))
        
        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
            return _in.reshape(-1, ch)
        
        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]
        
        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)
        
        boxes, classes, scores = self._filter_boxes(boxes, scores, classes_conf)
        
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self._nms_boxes(b, s)
            
            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])
        
        if not nclasses and not nscores:
            return None, None, None
        
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        
        return boxes, classes, scores