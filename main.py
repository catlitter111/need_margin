#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
瓶子检测与机器人控制主程序
整合双目相机瓶子检测、WebSocket通信和机器人控制功能
具有手动控制和自动采摘两种模式
"""

import cv2
import numpy as np
import time
import math
import json
import websocket
import threading
import psutil
import logging
import os
import base64
from queue import Queue, Empty
import random

# 导入自定义模块
from stereo_camera import StereoCamera
from robot_controller import RobotController, DIR_FORWARD, DIR_BACKWARD, DIR_LEFT, DIR_RIGHT, DIR_STOP
from servo_controller import ServoController, DEFAULT_SERVO_ID, CENTER_POSITION,SERVO_MODE
from bottle_detector import BottleDetector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# ====================配置参数====================
# WebSocket服务器配置
SERVER_URL = "ws://101.201.150.96:1234/ws/robot/robot_123"

# 摄像头配置
CAMERA_ID = 21  # 双目相机ID
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 480

# 质量预设配置
QUALITY_PRESETS = {
    "high": {
        "resolution": (640, 480),
        "fps": 15,
        "bitrate": 800,  # Kbps
        "quality": 80  # JPEG质量(1-100)
    },
    "medium": {
        "resolution": (480, 360),
        "fps": 10,
        "bitrate": 500,
        "quality": 70
    },
    "low": {
        "resolution": (320, 240),
        "fps": 8,
        "bitrate": 300,
        "quality": 60
    },
    "very_low": {
        "resolution": (240, 180),
        "fps": 5,
        "bitrate": 150,
        "quality": 50
    },
    "minimum": {
        "resolution": (160, 120),
        "fps": 3,
        "bitrate": 80,
        "quality": 40
    }
}
INITIAL_PRESET = "medium"  # 初始质量预设

# RKNN模型参数
RKNN_MODEL = "/home/elf/Desktop/project/new_project/yolo11n.rknn"
MODEL_SIZE = (640, 640)  # 模型输入尺寸

# 舵机控制相关的常量
DEFAULT_SERVO_PORT = "/dev/ttyS90"  # 默认舵机串口
DEFAULT_SERVO_BAUDRATE = 115200  # 默认波特率

# 机器人控制串口
ROBOT_SERIAL_PORT = "/dev/ttyS9"  # 机器人控制串口
ROBOT_SERIAL_BAUDRATE = 115200

# 全局状态变量
running = True
ws = None
connected = False
reconnect_count = 0
max_reconnect_attempts = 5
reconnect_interval = 3  # 重连间隔(秒)

# 视频配置
current_preset = INITIAL_PRESET
current_config = QUALITY_PRESETS[INITIAL_PRESET]
frame_queue = Queue(maxsize=10)  # 帧缓冲队列

# 手动/自动模式控制
operation_mode = "manual"  # 'manual'或'auto'
auto_harvest_active = False  # 自动采摘是否激活

# 机器人状态
robot_status = {
    "battery_level": 85,
    "position": {"x": 0, "y": 0, "latitude": 0.0, "longitude": 0.0},
    "harvested_count": 0,
    "cpu_usage": 0,
    "signal_strength": 70,
    "upload_bandwidth": 1000,  # 初始估计值(Kbps)
    "frames_sent": 0,
    "bytes_sent": 0,
    "last_bandwidth_check": time.time(),
    "last_bytes_sent": 0,
    "current_speed": 50,  # 默认速度为50%
    "current_direction": DIR_STOP,  # 默认方向为停止
    "working_hours": 0.0,  # 工作时间(小时)
    "working_area": 0.0,   # 工作面积(公顷)
    "total_harvested": 0,  # 总采摘量
    "today_harvested": 0   # 今日采摘量
}

# 瓶子检测结果
bottle_detections_with_distance = []  # 存储带距离信息的瓶子检测结果
nearest_bottle_distance = None  # 最近瓶子的距离

# 搜索相关变量
last_search_time = 0  # 上次搜索时间
last_detection_time = 0  # 上次检测到瓶子的时间

# 线程锁
video_lock = threading.Lock()

# ====================WebSocket客户端函数====================
def on_message(ws, message):
    global operation_mode, auto_harvest_active
    
    try:
        data = json.loads(message)
        message_type = data.get("type")

        if message_type == "command":
            handle_command(data)
        elif message_type == "quality_adjustment":
            handle_quality_adjustment(data)
        elif message_type == "set_position":
            handle_position_update(data)
        elif message_type == "mode_control":
            # 处理模式控制消息
            new_mode = data.get("mode")
            auto_harvest = data.get("harvest", False)
            
            if new_mode in ["manual", "auto"]:
                # 更新操作模式
                operation_mode = new_mode
                auto_harvest_active = auto_harvest
                
                # 停止机器人当前动作（安全措施）
                robot_controller.stop()
                robot_status["current_direction"] = DIR_STOP
                
                # 记录模式变更
                logger.info(f"切换到{new_mode}模式，自动采摘：{auto_harvest_active}")
                
                # 发送状态更新
                send_status_update()

    except json.JSONDecodeError:
        logger.error(f"收到无效JSON: {message}")
    except Exception as e:
        logger.error(f"处理消息错误: {e}")

def on_error(ws, error):
    logger.error(f"WebSocket错误: {error}")

def on_close(ws, close_status_code, close_msg):
    global connected
    connected = False
    logger.warning(f"WebSocket连接关闭: {close_status_code} {close_msg}")
    schedule_reconnect()

def on_open(ws):
    global connected, reconnect_count
    connected = True
    reconnect_count = 0
    logger.info("WebSocket连接已建立")

    # 发送初始状态
    send_status_update()

def schedule_reconnect():
    global reconnect_count

    if not running:
        return

    if reconnect_count < max_reconnect_attempts:
        reconnect_count += 1
        logger.info(f"计划第 {reconnect_count} 次重连，{reconnect_interval}秒后尝试...")

        time.sleep(reconnect_interval)

        if not connected and running:
            connect_to_server()
    else:
        logger.error(f"达到最大重连次数({max_reconnect_attempts})，停止重连")

def connect_to_server():
    global ws

    # 创建WebSocket连接
    ws = websocket.WebSocketApp(SERVER_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    # 在新线程中运行WebSocket连接
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()


# ====================命令处理函数====================
def handle_command(command_data):
    global robot_status
    
    cmd = command_data.get("command")
    params = command_data.get("params", {})
    speed = params.get("speed", robot_status["current_speed"])

    # 确保速度在有效范围内
    speed = max(0, min(100, speed))

    logger.info(f"收到命令: {cmd}, 参数: {params}")

    # 更新当前速度
    robot_status["current_speed"] = speed

    # 只有在手动模式下才执行这些命令
    if operation_mode == "manual":
        # 根据不同命令执行不同操作
        if cmd == "forward":
            robot_status["current_direction"] = DIR_FORWARD
            robot_controller.move(DIR_FORWARD, speed)
            logger.info(f"机器人前进, 速度: {speed}%")

        elif cmd == "backward":
            robot_status["current_direction"] = DIR_BACKWARD
            robot_controller.move(DIR_BACKWARD, speed)
            logger.info(f"机器人后退, 速度: {speed}%")

        elif cmd == "left":
            robot_status["current_direction"] = DIR_LEFT
            robot_controller.move(DIR_LEFT, speed)
            logger.info(f"机器人左转, 速度: {speed}%")

        elif cmd == "right":
            robot_status["current_direction"] = DIR_RIGHT
            robot_controller.move(DIR_RIGHT, speed)
            logger.info(f"机器人右转, 速度: {speed}%")

        elif cmd == "stop":
            robot_status["current_direction"] = DIR_STOP
            robot_controller.stop()
            logger.info("机器人停止")

        elif cmd == "set_motor_speed":
            # 单独设置速度命令
            robot_controller.set_speed(speed)
            logger.info(f"设置电机速度: {speed}%")

            # 如果当前有方向，更新该方向的速度
            if robot_status["current_direction"] != DIR_STOP:
                robot_controller.move(robot_status["current_direction"], speed)

        elif cmd == "startHarvest":
            # 采摘功能可以添加其他命令
            logger.info("开始采摘")
            # 模拟采摘增加
            robot_status["harvested_count"] += 1
            robot_status["today_harvested"] += 1
            robot_status["total_harvested"] += 1
            logger.info(f"采摘完成，当前总数: {robot_status['harvested_count']}")
            
            # 更新工作统计
            robot_status["working_hours"] += 0.02  # 增加约1分钟
            robot_status["working_area"] += 0.01  # 增加0.01公顷
            
            # 发送状态更新
            send_status_update()

        elif cmd == "stopHarvest":
            # 暂停采摘
            logger.info("暂停采摘")
    else:
        logger.info(f"忽略命令 {cmd}，当前处于自动模式")

    # 紧急停止命令在任何模式下都能执行
    if cmd == "emergencyStop":
        # 紧急停止所有功能
        robot_status["current_direction"] = DIR_STOP
        robot_controller.stop()
        logger.info("紧急停止所有操作")
        
        # 如果在自动模式，关闭自动采摘
        if operation_mode == "auto":
            auto_harvest_active = False
        
        # 发送状态更新
        send_status_update()


def handle_position_update(data):
    try:
        position_data = data.get("data", [])
        if len(position_data) >= 8:
            # 从字节数组中解析出经纬度
            lat_int = (position_data[0] << 24) | (position_data[1] << 16) | (position_data[2] << 8) | position_data[3]
            lon_int = (position_data[4] << 24) | (position_data[5] << 16) | (position_data[6] << 8) | position_data[7]

            # 处理有符号整数
            if lat_int & 0x80000000:
                lat_int = lat_int - 0x100000000
            if lon_int & 0x80000000:
                lon_int = lon_int - 0x100000000

            # 转换回浮点数
            latitude = lat_int / 1000000.0
            longitude = lon_int / 1000000.0

            # 更新机器人状态中的位置信息
            robot_status["position"]["latitude"] = latitude
            robot_status["position"]["longitude"] = longitude

            logger.info(f"收到位置更新: 纬度={latitude}, 经度={longitude}")

            # 发送位置更新到服务器
            if ws and connected:
                ws.send(json.dumps({
                    "type": "position_update",
                    "data": {
                        "latitude": latitude,
                        "longitude": longitude,
                        "timestamp": int(time.time() * 1000)
                    }
                }))
        else:
            logger.error("位置数据格式错误")
    except Exception as e:
        logger.error(f"处理位置更新错误: {e}")


def set_position(latitude, longitude):
    try:
        # 更新本地状态
        robot_status["position"]["latitude"] = latitude
        robot_status["position"]["longitude"] = longitude

        # 生成并发送位置命令到小车
        robot_controller.set_position(latitude, longitude)

        # 发送位置更新到服务器
        if ws and connected:
            ws.send(json.dumps({
                "type": "position_update",
                "data": {
                    "latitude": latitude,
                    "longitude": longitude,
                    "timestamp": int(time.time() * 1000)
                }
            }))

        logger.info(f"位置已设置: 纬度={latitude}, 经度={longitude}")
        return True
    except Exception as e:
        logger.error(f"设置位置失败: {e}")
        return False


def handle_quality_adjustment(adjustment_data):
    global current_preset, current_config

    preset = adjustment_data.get("preset")

    if preset in QUALITY_PRESETS:
        logger.info(f"收到质量调整命令: {preset}")

        # 更新当前质量设置
        current_preset = preset
        current_config = QUALITY_PRESETS[preset]

        # 发送调整结果
        if ws and connected:
            try:
                ws.send(json.dumps({
                    "type": "quality_adjustment_result",
                    "success": True,
                    "preset": preset,
                    "actual_resolution": f"{current_config['resolution'][0]}x{current_config['resolution'][1]}",
                    "actual_fps": current_config["fps"]
                }))
            except Exception as e:
                logger.error(f"发送质量调整结果失败: {e}")

        return True
    else:
        logger.error(f"未知的质量预设: {preset}")
        return False


# ====================图像显示函数====================
def draw_fps(image, fps):
    """在图像上绘制帧率信息"""
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def draw_mode_info(image, mode, auto_harvest):
    """在图像上绘制工作模式信息"""
    mode_str = "AUTO" if mode == "auto" and auto_harvest else ("MANUAL" if mode == "manual" else "STOP")
    cv2.putText(image, f"Mode: {mode_str}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def draw_servo_info(image, position):
    """在图像上绘制舵机位置信息"""
    cv2.putText(image, f"Position: {position}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def draw_auto_control_info(image, distance, robot_moving):
    """在图像上绘制自动控制信息"""
    if distance is not None:
        status = "moving" if robot_moving else "stopping"
        cv2.putText(image, f"Distance: {distance:.2f}m, Status: {status}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)


# ====================线程函数====================
def video_processing_thread(stereo_camera, bottle_detector, servo, robot):
    """视频处理线程：执行瓶子检测、舵机控制和自动控制"""
    global bottle_detections_with_distance, robot_status, nearest_bottle_distance
    global running, last_detection_time, last_search_time
    
    # 初始化变量
    current_servo_position = CENTER_POSITION
    start_time = time.time()
    frame_count = 0
    robot_moving = False
    last_detection_time = time.time()  # 上次检测到瓶子的时间
    last_search_time = time.time()     # 上次搜索时间
    has_bottle = False                 # 标记是否检测到瓶子
    
    while running:
        # 读取帧
        frame_left, frame_right = stereo_camera.capture_frame()
        if frame_left is None or frame_right is None:
            logger.warning("无法接收帧")
            time.sleep(0.1)
            continue
        
        # 校正左右相机图像
        frame_left_rectified, img_left_rectified, img_right_rectified = stereo_camera.rectify_stereo_images(frame_left, frame_right)
        
        # 计算视差
        disparity, disp_normalized = stereo_camera.compute_disparity(img_left_rectified, img_right_rectified)
        
        # 计算三维坐标
        threeD = stereo_camera.compute_3d_points(disparity)
        
        # 在左图上检测瓶子
        bottle_detections = bottle_detector.detect(frame_left)
        
        # 处理检测结果，计算距离
        local_bottle_detections_with_distance = []
        
        for left, top, right, bottom, score, cx, cy in bottle_detections:
            # 获取瓶子中心点的3D坐标和距离
            distance = stereo_camera.get_bottle_distance(threeD, cx, cy)
            
            if distance is not None:
                logger.debug(f'瓶子检测: 坐标 [{left}, {top}, {right}, {bottom}], 分数: {score:.2f}, 距离: {distance:.2f}m')
                # 在图像上绘制瓶子和距离信息
                bottle_detector.draw_detection(frame_left, (left, top, right, bottom, score), distance)
                # 添加到带距离信息的瓶子检测结果
                local_bottle_detections_with_distance.append((left, top, right, bottom, score, distance, cx, cy))
            else:
                # 如果无法计算距离，仍然绘制瓶子但不显示距离
                bottle_detector.draw_detection(frame_left, (left, top, right, bottom, score))
        
        # 更新全局的瓶子检测结果
        bottle_detections_with_distance = local_bottle_detections_with_distance
        
        # 检查是否检测到瓶子
        current_time = time.time()
        if local_bottle_detections_with_distance:
            has_bottle = True
            last_detection_time = current_time
            
            # 找出距离最近的瓶子
            nearest_bottle = min(local_bottle_detections_with_distance, key=lambda x: x[5])
            _, _, _, _, _, distance, cx, _ = nearest_bottle
            nearest_bottle_distance = distance
            
            # 控制舵机跟踪最近的瓶子
            if servo.serial:
                # 跟踪瓶子
                current_servo_position = servo.track_object(
                    frame_left.shape[1],  # 图像宽度
                    cx,
                    DEFAULT_SERVO_ID,
                    current_servo_position
                )
                
                # 更新最近瓶子的距离
        else:
            # 如果超过2秒未检测到瓶子，停止舵机
            if has_bottle and (current_time - last_detection_time) > 2.0:
                has_bottle = False
                # 停止舵机
                if servo.serial:
                    servo.stop_servo(DEFAULT_SERVO_ID)
                    logger.info("未检测到瓶子，舵机已停止")
                nearest_bottle_distance = None
        
        # 自动模式下控制机器人
        if operation_mode == "auto" and auto_harvest_active:
            # 根据检测到的瓶子状态控制机器人
            
            # 有检测到瓶子
            if nearest_bottle_distance is not None:
                if nearest_bottle_distance > 0.8:  # 距离大于1米
                    if not robot_moving:
                        # 控制机器人向前移动
                        robot.move(DIR_FORWARD, robot_status["current_speed"])
                        robot_status["current_direction"] = DIR_FORWARD
                        robot_moving = True
                        logger.info(f"自动模式: 瓶子距离 {nearest_bottle_distance:.2f}m > 1.0m，机器人开始前进")
                        
                        # 根据瓶子的位置适当调整方向
                        # 找到最近的瓶子中心点
                        if bottle_detections_with_distance:
                            nearest_bottle = min(bottle_detections_with_distance, key=lambda x: x[5])
                            _, _, _, _, _, _, cx, _ = nearest_bottle
                            
                            # 计算图像中心
                            image_center = frame_left.shape[1] // 2
                            # 计算偏移量（以像素为单位）
                            offset = cx - image_center
                            
                            # 如果偏离中心过多，调整方向
                            if abs(offset) > frame_left.shape[1] // 6:  # 偏离超过1/6宽度
                                if offset > 0:  # 瓶子在右侧
                                    robot.move(DIR_RIGHT, max(30, robot_status["current_speed"] - 20))
                                    logger.info("自动模式: 目标偏右，向右调整")
                                else:  # 瓶子在左侧
                                    robot.move(DIR_LEFT, max(30, robot_status["current_speed"] - 20))
                                    logger.info("自动模式: 目标偏左，向左调整")
                else:
                    if robot_moving:
                        # 停止机器人
                        robot.stop()
                        robot_status["current_direction"] = DIR_STOP
                        robot_moving = False
                        logger.info(f"自动模式: 瓶子距离 {nearest_bottle_distance:.2f}m <= 1.0m，机器人停止，准备采摘")
                        
                        # 模拟采摘动作 - 这里应该增加实际的采摘动作代码
                        if current_time - last_detection_time > 1.0:  # 避免重复触发
                            # 更新采摘计数
                            robot_status["harvested_count"] = robot_status.get("harvested_count", 0) + 1
                            robot_status["today_harvested"] = robot_status.get("today_harvested", 0) + 1
                            robot_status["total_harvested"] = robot_status.get("total_harvested", 0) + 1
                            logger.info(f"自动模式: 完成一次采摘，当前总数: {robot_status['harvested_count']}")
                            
                            # 更新工作时间和面积
                            robot_status["working_hours"] = robot_status.get("working_hours", 0) + 0.05  # 增加3分钟
                            robot_status["working_area"] = robot_status.get("working_area", 0) + 0.02  # 增加0.02公顷
            else:
                # 没有检测到瓶子，但正在移动，应该停止并搜索
                if robot_moving:
                    robot.stop()
                    robot_status["current_direction"] = DIR_STOP
                    robot_moving = False
                    logger.info("自动模式: 未检测到瓶子，机器人停止")
                elif current_time - last_detection_time > 5.0:
                    # 超过5秒未检测到瓶子，开始随机搜索
                    # 每5秒改变一次搜索方向
                    # if int(current_time) % 5 == 0 and int(current_time) != int(last_search_time):
                    #     last_search_time = current_time
                    #     # 随机选择一个方向
                    #     search_directions = [DIR_FORWARD, DIR_LEFT, DIR_RIGHT]
                    #     random_dir = random.choice(search_directions)
                        
                    #     # 如果选择了向前，设置较高速度
                    #     speed = robot_status["current_speed"] if random_dir == DIR_FORWARD else max(30, robot_status["current_speed"] - 20)
                        
                    #     # 移动指定方向2秒
                    #     robot.move(random_dir, speed)
                    #     robot_status["current_direction"] = random_dir
                    #     robot_moving = True
                    #     logger.info(f"自动模式: 搜索中，随机选择方向: {random_dir}")
                        
                    #     # 2秒后停止，等待新的检测结果
                    #     time.sleep(2)
                    #     robot.stop()
                    #     robot_status["current_direction"] = DIR_STOP
                    #     robot_moving = False

        elif operation_mode == "auto" and not auto_harvest_active and robot_moving:
            # 如果自动采摘未激活但机器人在移动，停止机器人
            robot.stop()
            robot_status["current_direction"] = DIR_STOP
            robot_moving = False
            logger.info("自动模式未激活，机器人停止")
        
        # 在图像上显示舵机位置信息
        draw_servo_info(frame_left, current_servo_position)
        
        # 在图像上显示模式信息
        draw_mode_info(frame_left, operation_mode, auto_harvest_active)
        
        # 在自动模式下显示控制信息
        if operation_mode == "auto" and auto_harvest_active:
            draw_auto_control_info(frame_left, nearest_bottle_distance, robot_moving)
        
        # 计算并显示帧率
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        draw_fps(frame_left, fps)
        
        # 将处理后的帧放入队列用于发送到服务器
        try:
            # 根据当前配置调整图像大小
            resized_frame = cv2.resize(frame_left, current_config["resolution"])
            # 非阻塞方式，如果队列满了就丢弃帧
            if not frame_queue.full():
                frame_queue.put_nowait(resized_frame)
        except Exception as e:
            logger.error(f"放入帧队列失败: {e}")
        
        # 显示结果
        with video_lock:
            # cv2.imshow("Origin Left", frame_left)
            # cv2.imshow("Origin Right", frame_right)
            cv2.imshow("Bottle Detect", frame_left)
            # cv2.imshow("SVGM", disp_normalized)
        
        # 按Q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break


def status_update_thread():
    """状态更新线程：定期发送机器人状态到服务器"""
    last_status_time = 0
    status_interval = 5  # 每5秒发送一次状态

    while running:
        try:
            current_time = time.time()

            # 更新CPU使用率
            robot_status["cpu_usage"] = psutil.cpu_percent(interval=0.1)

            # 计算上传带宽
            bytes_sent_diff = robot_status["bytes_sent"] - robot_status["last_bytes_sent"]
            time_diff = current_time - robot_status["last_bandwidth_check"]

            if time_diff > 0:
                # 计算Kbps
                upload_speed = (bytes_sent_diff * 8) / (time_diff * 1000)
                # 平滑带宽估计
                robot_status["upload_bandwidth"] = (robot_status["upload_bandwidth"] * 0.7) + (upload_speed * 0.3)

                # 更新检查点
                robot_status["last_bandwidth_check"] = current_time
                robot_status["last_bytes_sent"] = robot_status["bytes_sent"]

            # 定期发送状态更新
            if current_time - last_status_time >= status_interval:
                send_status_update()
                last_status_time = current_time

            time.sleep(1)  # 每秒更新一次CPU和带宽，但5秒才发送一次状态

        except Exception as e:
            logger.error(f"状态更新错误: {e}")


def send_status_update():
    """发送状态更新到服务器"""
    if ws and connected:
        try:
            # 生成当前机器人的详细状态
            today = time.strftime("%Y-%m-%d")
            
            # 创建更详细的状态数据
            detailed_status = {
                "type": "status_update",
                "data": {
                    # 基本数据
                    "battery_level": robot_status["battery_level"],
                    "cpu_usage": robot_status["cpu_usage"],
                    "signal_strength": robot_status["signal_strength"],
                    "upload_bandwidth": robot_status["upload_bandwidth"],
                    "frames_sent": robot_status["frames_sent"],
                    "bytes_sent": robot_status["bytes_sent"],
                    
                    # 视频和网络相关
                    "current_preset": current_preset,
                    
                    # 运动控制相关
                    "current_speed": robot_status["current_speed"],
                    "current_direction": robot_status["current_direction"],
                    "position": robot_status["position"],
                    
                    # 模式和瓶子检测相关
                    "operation_mode": operation_mode,
                    "auto_harvest_active": auto_harvest_active,
                    "nearest_bottle_distance": nearest_bottle_distance,
                    
                    # 采摘统计
                    "today_harvested": robot_status["today_harvested"],
                    "total_harvested": robot_status["total_harvested"],
                    "working_hours": round(robot_status["working_hours"], 1),
                    "working_area": round(robot_status["working_area"], 1),
                    
                    # 添加适合界面显示的状态文本
                    "status": get_status_text()
                }
            }
            
            # 发送到服务器
            ws.send(json.dumps(detailed_status))
            logger.debug("状态更新已发送")
        except Exception as e:
            logger.error(f"发送状态更新失败: {e}")


def get_status_text():
    """根据当前状态生成适合界面显示的状态文本"""
    if operation_mode == "auto":
        if auto_harvest_active:
            if nearest_bottle_distance is not None:
                if nearest_bottle_distance <= 1.0:
                    return "已到达采摘位置"
                else:
                    return "自动采摘中"
            else:
                return "搜寻采摘目标"
        else:
            return "自动模式已就绪"
    else:  # manual mode
        if robot_status["current_direction"] == DIR_STOP:
            return "待命中"
        else:
            direction_text = {
                DIR_FORWARD: "前进",
                DIR_BACKWARD: "后退",
                DIR_LEFT: "左转",
                DIR_RIGHT: "右转"
            }.get(robot_status["current_direction"], "移动中")
            return f"手动控制 - {direction_text}"


def video_sending_thread():
    """视频发送线程：将视频帧发送到WebSocket服务器"""
    global robot_status

    last_frame_time = time.time()
    frame_interval = 1.0 / QUALITY_PRESETS[INITIAL_PRESET]["fps"]

    # 用于计算实际FPS
    fps_counter = 0
    fps_timer = time.time()
    actual_fps = 0

    while running:
        try:
            # 控制发送频率
            current_time = time.time()
            elapsed = current_time - last_frame_time

            # 根据当前预设更新帧间隔
            frame_interval = 1.0 / current_config["fps"]

            if elapsed < frame_interval:
                time.sleep(0.001)  # 短暂休眠，减少CPU使用
                continue

            # 尝试从队列获取一帧
            try:
                frame = frame_queue.get(block=False)
            except Empty:
                time.sleep(0.01)
                continue

            # 处理并发送帧
            if ws and connected:
                try:
                    # 调整大小以匹配当前分辨率设置
                    if frame.shape[1] != current_config["resolution"][0] or frame.shape[0] != \
                            current_config["resolution"][1]:
                        frame = cv2.resize(frame, current_config["resolution"])

                    # 压缩帧为JPEG格式
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), current_config["quality"]]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)

                    # 转为Base64编码
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                    # 构建消息
                    message = {
                        "type": "video_frame",
                        "preset": current_preset,
                        "resolution": f"{current_config['resolution'][0]}x{current_config['resolution'][1]}",
                        "timestamp": int(current_time * 1000),  # 毫秒时间戳
                        "data": jpg_as_text
                    }

                    # 发送视频帧
                    message_json = json.dumps(message)
                    ws.send(message_json)

                    # 更新统计信息
                    robot_status["frames_sent"] += 1
                    robot_status["bytes_sent"] += len(message_json)

                    # 计算实际FPS
                    fps_counter += 1
                    if current_time - fps_timer >= 1.0:  # 每秒计算一次
                        actual_fps = fps_counter
                        fps_counter = 0
                        fps_timer = current_time
                        logger.debug(f"实际FPS: {actual_fps}, 目标FPS: {current_config['fps']}")

                    # 更新最后发送时间
                    last_frame_time = current_time

                except Exception as e:
                    logger.error(f"处理视频帧错误: {e}")

        except Exception as e:
            logger.error(f"视频发送线程错误: {e}")


def battery_simulation_thread():
    """电池模拟线程：模拟电池电量变化"""
    while running:
        try:
            # 缓慢减少电池电量
            robot_status["battery_level"] = max(0, robot_status["battery_level"] - 0.02)

            # 随机变化信号强度
            if random.random() < 0.1:  # 10%的概率改变信号强度
                signal_delta = random.uniform(-5, 5)
                robot_status["signal_strength"] = max(0, min(100, robot_status["signal_strength"] + signal_delta))

            time.sleep(5)

        except Exception as e:
            logger.error(f"电池模拟错误: {e}")


# ====================主函数====================
def main():
    global running, robot_controller
    
    try:
        logger.info("启动瓶子检测与机器人控制集成程序...")
        
        # 初始化双目相机
        logger.info('--> 初始化双目相机')
        stereo_camera = StereoCamera(CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT)
        if not stereo_camera.open_camera():
            logger.error("无法打开相机，程序退出")
            return
        
        # 加载相机参数
        logger.info('--> 加载相机参数')
        stereo_camera.load_camera_params('/home/elf/Desktop/project/new_project/out.xls')
        
        # 设置双目校正参数
        logger.info('--> 设置双目校正参数')
        stereo_camera.setup_stereo_rectification()
        
        # 初始化舵机控制器
        logger.info('--> 初始化舵机控制器')
        servo = ServoController(DEFAULT_SERVO_PORT, DEFAULT_SERVO_BAUDRATE)
        if not servo.serial:
            logger.error("舵机初始化失败，程序将继续运行但不会控制舵机")
        else:
            # 设置舵机为180度顺时针模式
            servo.set_mode(DEFAULT_SERVO_ID, SERVO_MODE)
            # 将舵机移动到中心位置
            servo.center_servo(DEFAULT_SERVO_ID)
            servo.set_initial_position()
            time.sleep(1)
            logger.info("舵机已设置为180度顺时针模式并居中")
        
        # 初始化机器人控制器
        logger.info('--> 初始化机器人控制器')
        robot_controller = RobotController(ROBOT_SERIAL_PORT, ROBOT_SERIAL_BAUDRATE)
        
        # 初始化瓶子检测器
        logger.info('--> 初始化瓶子检测器')
        bottle_detector = BottleDetector(RKNN_MODEL, MODEL_SIZE)
        if not bottle_detector.load_model():
            logger.error("加载瓶子检测模型失败，程序退出")
            return
        
        # 连接到WebSocket服务器
        logger.info('--> 连接到WebSocket服务器')
        connect_to_server()
        
        # 创建并启动状态更新线程
        status_thread = threading.Thread(target=status_update_thread)
        status_thread.daemon = True
        status_thread.start()
        
        # 创建并启动视频发送线程
        sending_thread = threading.Thread(target=video_sending_thread)
        sending_thread.daemon = True
        sending_thread.start()
        
        # 创建并启动电池模拟线程
        battery_thread = threading.Thread(target=battery_simulation_thread)
        battery_thread.daemon = True
        battery_thread.start()
        
        # 设置初始位置 (北京天安门广场坐标)
        set_position(39.9042, 116.4074)
        
        # 主线程执行视频处理
        logger.info('--> 开始视频处理')
        video_processing_thread(stereo_camera, bottle_detector, servo, robot_controller)
        
    except KeyboardInterrupt:
        logger.info("接收到终止信号，正在关闭...")
    except Exception as e:
        logger.error(f"主线程错误: {e}")
    finally:
        running = False
        
        # 关闭WebSocket连接
        if ws:
            ws.close()
        
        # 关闭机器人控制串口
        if 'robot_controller' in globals() and robot_controller.serial:
            robot_controller.stop()  # 停止机器人
            robot_controller.disconnect()
        
        # 重置舵机位置并断开连接
        if 'servo' in locals() and servo.serial:
            servo.center_servo(DEFAULT_SERVO_ID)
            servo.disconnect()
        
        # 释放瓶子检测器资源
        if 'bottle_detector' in locals():
            bottle_detector.release_model()
        
        # 关闭相机
        if 'stereo_camera' in locals():
            stereo_camera.close_camera()
        
        # 关闭所有窗口
        cv2.destroyAllWindows()
        
        logger.info("程序已安全退出")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序异常退出: {e}")


