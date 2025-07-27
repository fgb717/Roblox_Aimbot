import sys
import time
import cv2
import numpy as np
import yaml
import onnxruntime as ort
import dxcam
import math
import ctypes
import threading
from collections import deque
from pynput import keyboard, mouse
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import QRect, Qt, QTimer, QPoint
from PyQt5.QtGui import QPainter, QFont, QImage, QColor, QPen, QBrush

class SafeMouseController:
    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.INPUT_MOUSE = 0
        self.MOUSEEVENTF_MOVE = 0x0001
        self.last_move_time = 0
        self.move_lock = threading.Lock()
        
    def move_relative(self, dx, dy):
        with self.move_lock:
            current_time = time.perf_counter()
            if current_time - self.last_move_time < 0.002:  # 500Hz移动频率
                return
                
            max_step = 30  # 增大单次最大移动量
            dx = max(-max_step, min(max_step, dx))
            dy = max(-max_step, min(max_step, dy))
            
            if dx == 0 and dy == 0:
                return
                
            class MOUSEINPUT(ctypes.Structure):
                _fields_ = [
                    ("dx", ctypes.c_long),
                    ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong),
                    ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
                ]
            
            class INPUT(ctypes.Structure):
                _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]
            
            mi = MOUSEINPUT(dx, dy, 0, self.MOUSEEVENTF_MOVE, 0, None)
            input_struct = INPUT(self.INPUT_MOUSE, mi)
            
            try:
                self.user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(input_struct))
                self.last_move_time = current_time
            except Exception as e:
                print(f"Mouse move error: {e}")

class StableTargetPredictor:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        # 定义检测区域 (316, 122) 到 (1689, 884)
        self.detection_x1 = 316
        self.detection_y1 = 122
        self.detection_x2 = 1689
        self.detection_y2 = 884
        self.detection_width = self.detection_x2 - self.detection_x1
        self.detection_height = self.detection_y2 - self.detection_y1
        
        self.prediction_time = 0.04  # 固定预测时间150ms
        self.smooth_factor = 0.6
        self.max_speed = 2000
        self.target_history = {}
        self.next_id = 0
        self.mouse_sensitivity = 1  # 提高默认灵敏度
        self.max_mouse_step = 30
        self.min_move_threshold = 2
        
        # PID控制参数
        self.pid_kp = 0.1  # 比例系数
        self.pid_ki = 0.0000001 # 积分系数
        self.pid_kd = 0   # 微分系数
        self.pid_integral_x = 0
        self.pid_integral_y = 0
        self.pid_last_error_x = 0
        self.pid_last_error_y = 0
        self.pid_last_time = time.perf_counter()
        
        # 方向键偏移参数
        self.aim_offset_x = 0  # 水平偏移量
        self.aim_offset_y = 0  # 垂直偏移量
        self.max_aim_offset = 50  # 最大偏移量
        self.offset_step = 5     # 每次按键偏移步长
        
    def update_aim_offset(self, key, pressed):
        """根据按键更新瞄准点偏移"""
        if key == keyboard.Key.up:
            self.aim_offset_y = -self.max_aim_offset if pressed else 0
        elif key == keyboard.Key.down:
            self.aim_offset_y = self.max_aim_offset if pressed else 0
        elif key == keyboard.Key.left:
            self.aim_offset_x = -self.max_aim_offset if pressed else 0
        elif key == keyboard.Key.right:
            self.aim_offset_x = self.max_aim_offset if pressed else 0
    
    def update_targets(self, detections, current_time):
        active_ids = set()
        updated_targets = []
        
        for det in detections:
            x1, y1, x2, y2, score, class_id = det
            width = x2 - x1
            height = y2 - y1
            
            # 修正中心点计算 - 瞄准胸部/头部区域
            center_x = x1 + width * 0.5  # 水平居中
            center_y = y1 + height * 0.3  # 垂直方向30%处
            
            matched_id = self._match_target(center_x, center_y, width, height)
            
            if matched_id is not None:
                pred_x, pred_y = self._update_target(
                    matched_id, x1, y1, x2, y2, score, class_id, current_time)
                active_ids.add(matched_id)
            else:
                matched_id = self._add_new_target(
                    x1, y1, x2, y2, score, class_id, current_time)
                pred_x, pred_y = center_x, center_y
                active_ids.add(matched_id)
            
            updated_targets.append({
                'id': matched_id,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'pred_x': pred_x, 'pred_y': pred_y,
                'score': score, 'class_id': class_id,
                'center_x': center_x, 'center_y': center_y
            })
        
        self._cleanup_stale_targets(active_ids, current_time)
        return updated_targets
    
    def _match_target(self, x, y, width, height):
        best_match = None
        min_distance = float('inf')
        match_threshold = max(width, height) * 0.7
        
        for target_id, history in self.target_history.items():
            if not history:
                continue
                
            last = history[-1]
            last_x = last['center_x']
            last_y = last['center_y']
            
            distance = math.sqrt((x - last_x)**2 + (y - last_y)**2)
            
            if distance < min_distance and distance < match_threshold:
                min_distance = distance
                best_match = target_id
                
        return best_match
    
    def _update_target(self, target_id, x1, y1, x2, y2, score, class_id, current_time):
        history = self.target_history[target_id]
        last = history[-1]
        
        dt = max(0.016, current_time - last['time'])  # 最小16ms
        
        # 使用低通滤波器平滑速度
        current_center_x = x1 + (x2 - x1) * 0.5
        current_center_y = y1 + (y2 - y1) * 0.3
        
        dx = (current_center_x - last['center_x']) / dt
        dy = (current_center_y - last['center_y']) / dt
        
        if 'dx' in last:
            dx = dx * 0.7 + last['dx'] * 0.3
            dy = dy * 0.7 + last['dy'] * 0.3
        
        speed = math.sqrt(dx**2 + dy**2)
        if speed > self.max_speed:
            dx = dx * self.max_speed / speed
            dy = dy * self.max_speed / speed
        
        pred_x = current_center_x + dx * self.prediction_time
        pred_y = current_center_y + dy * self.prediction_time
        
        # 确保预测点不超出检测区域
        pred_x = max(self.detection_x1, min(self.detection_x2, pred_x))
        pred_y = max(self.detection_y1, min(self.detection_y2, pred_y))
        
        history.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'score': score, 'class_id': class_id,
            'time': current_time,
            'dx': dx, 'dy': dy,
            'pred_x': pred_x, 'pred_y': pred_y,
            'center_x': current_center_x, 'center_y': current_center_y
        })
        
        if len(history) > 5:
            history.popleft()
            
        return pred_x, pred_y
    
    def _add_new_target(self, x1, y1, x2, y2, score, class_id, current_time):
        target_id = self.next_id
        self.next_id += 1
        
        center_x = x1 + (x2 - x1) * 0.5
        center_y = y1 + (y2 - y1) * 0.3
        
        self.target_history[target_id] = deque([{
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'score': score, 'class_id': class_id,
            'time': current_time,
            'dx': 0, 'dy': 0,
            'pred_x': center_x, 'pred_y': center_y,
            'center_x': center_x, 'center_y': center_y
        }], maxlen=5)
        
        return target_id
    
    def _cleanup_stale_targets(self, active_ids, current_time):
        stale_ids = [tid for tid, hist in self.target_history.items() 
                    if tid not in active_ids and 
                    (not hist or current_time - hist[-1]['time'] > 0.5)]
        for tid in stale_ids:
            del self.target_history[tid]
    
    def calculate_aim_offset(self, target, mouse_x, mouse_y):
        """使用PID控制计算瞄准偏移"""
        current_time = time.perf_counter()
        dt = max(0.001, current_time - self.pid_last_time)
        self.pid_last_time = current_time
        
        # 计算目标中心点 (根据目标大小调整偏移百分比)
        target_width = target['x2'] - target['x1']
        target_height = target['y2'] - target['y1']
        
        # 根据方向键偏移计算最终瞄准点 (偏移量基于目标大小的百分比)
        offset_factor_x = 0.2 * (target_width / self.detection_width)
        offset_factor_y = 0.15 * (target_height / self.detection_height)
        
        aim_x = target['pred_x'] + self.aim_offset_x * offset_factor_x
        aim_y = target['pred_y'] + self.aim_offset_y * offset_factor_y
        
        # 计算误差
        error_x = aim_x - mouse_x
        error_y = aim_y - mouse_y
        
        # PID计算
        self.pid_integral_x += error_x * dt
        self.pid_integral_y += error_y * dt
        
        derivative_x = (error_x - self.pid_last_error_x) / dt
        derivative_y = (error_y - self.pid_last_error_y) / dt
        
        output_x = (self.pid_kp * error_x + 
                   self.pid_ki * self.pid_integral_x + 
                   self.pid_kd * derivative_x)
        
        output_y = (self.pid_kp * error_y + 
                   self.pid_ki * self.pid_integral_y + 
                   self.pid_kd * derivative_y)
        
        self.pid_last_error_x = error_x
        self.pid_last_error_y = error_y
        
        # 限制输出
        distance = math.sqrt(output_x**2 + output_y**2)
        if distance < self.min_move_threshold:
            return 0, 0
            
        output_x *= self.mouse_sensitivity
        output_y *= self.mouse_sensitivity
        
        if distance > self.max_mouse_step:
            output_x = output_x * self.max_mouse_step / distance
            output_y = output_y * self.max_mouse_step / distance
            
        return int(output_x), int(output_y)

class YOLOv5_DML:
    def __init__(self):
        self.config = None
        self.session = None
        self.classes = ["object"]
        
    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            required_fields = ['model_path', 'input_width', 'input_height',
                             'confidence_threshold', 'nms_threshold', 'preprocess']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing config field: {field}")
                    
            config['classes'] = config.get('classes', ["object"])
            self.config = config
            self.classes = config['classes']
            return True
        except Exception as e:
            QMessageBox.critical(None, "Config Error", f"Failed to load config:\n{str(e)}")
            return False
    
    def init_session(self):
        if not self.config:
            return False
            
        try:
            providers = [('DmlExecutionProvider', {'device_id': 0})]
            self.session = ort.InferenceSession(
                self.config['model_path'],
                providers=providers
            )
            return True
        except Exception as e:
            QMessageBox.critical(None, "Init Error", f"Failed to initialize DML:\n{str(e)}")
            return False
    
    def preprocess(self, image):
        # 裁剪图像到检测区域 (316,122)-(1689,884)
        cropped = image[122:884, 316:1689]
        img = cv2.resize(cropped, (self.config['input_width'], self.config['input_height']))
        if self.config['preprocess']['swap_rgb']:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) * self.config['preprocess']['scale']
        return np.transpose(img, (2, 0, 1))[np.newaxis]
    
    def detect(self, image):
        if not self.session:
            return []
            
        blob = self.preprocess(image)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        
        predictions = np.squeeze(outputs[0])
        if len(predictions.shape) == 1:
            predictions = np.expand_dims(predictions, 0)
            
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        
        if predictions.shape[1] > 5:
            class_ids = np.argmax(predictions[:, 5:], axis=1)
            scores *= np.max(predictions[:, 5:], axis=1)
        else:
            class_ids = np.zeros(len(scores))
        
        valid = scores > self.config['confidence_threshold']
        boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
        
        if len(scores) == 0:
            return []
        
        # 将坐标映射回裁剪区域
        boxes[:, [0, 2]] *= (1689-316) / self.config['input_width']
        boxes[:, [1, 3]] *= (884-122) / self.config['input_height']
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        
        # 将坐标映射回原始图像
        boxes[:, [0, 2]] += 316
        boxes[:, [1, 3]] += 122
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                                 self.config['confidence_threshold'],
                                 self.config['nms_threshold'])
        
        if len(indices) == 0:
            return []
            
        return np.column_stack([boxes[indices], scores[indices], class_ids[indices]])

class AimOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        screen = QApplication.primaryScreen()
        self.setGeometry(0, 0, screen.size().width(), screen.size().height())
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        self.targets = []
        self.classes = ["object"]
        self.stats = {"fps": 0, "inference": 0, "sensitivity": 1.2}
        self.aim_assist_enabled = False
        self.aim_assist_active = False
        self.show_thumbnail = True
        self.detection_rect = QRect(316, 122, 1689-316, 884-122)  # 检测区域矩形
        
        # 缩略图参数
        self.thumbnail_scale = 0.3
        self.thumbnail_width = int(1000 * self.thumbnail_scale)
        self.thumbnail_height = int(500 * self.thumbnail_scale)
        self.thumbnail_pos = QPoint(10, 10)
        
        # 存储当前帧
        self.current_frame = None
    
    def update_overlay(self, frame, targets, stats, aim_assist_enabled, aim_assist_active):
        self.current_frame = frame
        self.targets = targets
        self.stats = stats
        self.aim_assist_enabled = aim_assist_enabled
        self.aim_assist_active = aim_assist_active
        self.update()
    
    def paintEvent(self, event):
        if self.current_frame is None or not self.show_thumbnail:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. 绘制检测区域边框
        painter.setPen(QPen(QColor(255, 0, 0, 150), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self.detection_rect)
        
        # 2. 绘制缩略图背景
        h, w = self.current_frame.shape[:2]
        bytes_per_line = 3 * w
        qimage = QImage(
            self.current_frame.data, w, h, bytes_per_line, 
            QImage.Format_BGR888
        )
        
        scaled_w = self.thumbnail_width
        scaled_h = int(h * (scaled_w / w))
        if scaled_h > self.thumbnail_height:
            scaled_h = self.thumbnail_height
            scaled_w = int(w * (scaled_h / h))
        
        # 绘制半透明背景
        bg_padding = 5
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 180), Qt.SolidPattern))
        painter.drawRect(
            self.thumbnail_pos.x() - bg_padding,
            self.thumbnail_pos.y() - bg_padding,
            scaled_w + 2 * bg_padding,
            scaled_h + 2 * bg_padding
        )
        
        # 绘制原画面
        painter.drawImage(
            QRect(
                self.thumbnail_pos.x(), 
                self.thumbnail_pos.y(),
                scaled_w,
                scaled_h
            ),
            qimage
        )
        
        # 3. 在缩略图上绘制目标信息
        painter.save()
        scale_x = scaled_w / w
        scale_y = scaled_h / h
        
        painter.translate(self.thumbnail_pos.x(), self.thumbnail_pos.y())
        painter.scale(scale_x, scale_y)
        
        for target in self.targets:
            x1, y1, x2, y2 = target['x1'], target['y1'], target['x2'], target['y2']
            center_x, center_y = target['center_x'], target['center_y']
            pred_x, pred_y = target['pred_x'], target['pred_y']
            score = target['score']
            class_id = int(target['class_id'])
            
            # 绘制检测框
            hue = int(179 * class_id / max(1, len(self.classes)-1))
            color = QColor.fromHsv(hue, 255, 255, 150)
            painter.setPen(QPen(color, max(1, int(2 / scale_x))))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            
            # 绘制修正后的瞄准点
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
            painter.drawEllipse(QPoint(int(center_x), int(center_y)), 
                              max(2, int(3 / scale_x)), 
                              max(2, int(3 / scale_y)))
            
            # 绘制预测点
            painter.setPen(QPen(Qt.red, max(1, int(2 / scale_x))))
            cross_size = max(5, int(12 * min(scale_x, scale_y)))
            painter.drawLine(
                int(pred_x - cross_size), int(pred_y),
                int(pred_x + cross_size), int(pred_y)
            )
            painter.drawLine(
                int(pred_x), int(pred_y - cross_size),
                int(pred_x), int(pred_y + cross_size)
            )
            
            # 绘制从中心到预测点的箭头
            arrow_size = max(3, int(8 * min(scale_x, scale_y)))
            angle = math.atan2(pred_y - center_y, pred_x - center_x)
            p1 = QPoint(
                int(pred_x - arrow_size * math.cos(angle - math.pi/6)),
                int(pred_y - arrow_size * math.sin(angle - math.pi/6))
            )
            p2 = QPoint(
                int(pred_x - arrow_size * math.cos(angle + math.pi/6)),
                int(pred_y - arrow_size * math.sin(angle + math.pi/6))
            )
            
            painter.setPen(QPen(Qt.yellow, max(1, int(1 / scale_x))))
            painter.drawLine(int(center_x), int(center_y), int(pred_x), int(pred_y))
            painter.drawLine(int(pred_x), int(pred_y), p1.x(), p1.y())
            painter.drawLine(int(pred_x), int(pred_y), p2.x(), p2.y())
            
            # 绘制标签
            if score > 0.3 and class_id < len(self.classes):
                label = f"{self.classes[class_id]}: {score:.2f}"
                font = QFont("Arial", max(8, int(10 * min(scale_x, scale_y))))
                painter.setFont(font)
                text_rect = painter.fontMetrics().boundingRect(label)
                
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(0, 0, 0, 180), Qt.SolidPattern))
                painter.drawRect(
                    x1 + 2, y1 + 2, 
                    text_rect.width() + 4, 
                    text_rect.height() + 4
                )
                
                painter.setPen(QPen(Qt.white, 1))
                painter.drawText(
                    x1 + 5, 
                    y1 + text_rect.height() + 5, 
                    label
                )
        
        painter.restore()
        
        # 4. 在缩略图下方显示状态信息
        status_text = f"Aim Assist: {'ON' if self.aim_assist_enabled else 'OFF'}"
        if self.aim_assist_enabled:
            status_text += f" | Sens: {self.stats['sensitivity']:.1f}"
        
        info_text = f"FPS: {self.stats['fps']:.1f} | Infer: {self.stats['inference']*1000:.1f}ms"
        
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        
        text_x = self.thumbnail_pos.x()
        text_y = self.thumbnail_pos.y() + scaled_h + 25
        
        bg_width = max(
            metrics.width(status_text),
            metrics.width(info_text)
        ) + 20
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 150), Qt.SolidPattern))
        painter.drawRect(
            text_x - 5, 
            text_y - metrics.height() - 5,
            bg_width,
            metrics.height() * 2 + 10
        )
        
        painter.setPen(QPen(Qt.white, 1))
        painter.drawText(text_x, text_y, status_text)
        painter.drawText(text_x, text_y + metrics.height(), info_text)

class AimAssistSystem:
    def __init__(self):
        self.screen_width, self.screen_height = self._get_screen_size()
        self.mouse_controller = SafeMouseController()
        self.predictor = StableTargetPredictor(self.screen_width, self.screen_height)
        self.detector = YOLOv5_DML()
        self.overlay = AimOverlay()
        
        self.aim_assist_enabled = True
        self.aim_assist_active = False
        self.left_mouse_pressed = False
        self.right_mouse_pressed = False
        
        # 输入监听
        self.mouse_listener = mouse.Listener(on_click=self._on_mouse_click)
        self.key_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        
        self.sensitivity_step = 0.1

    def _get_screen_size(self):
        app = QApplication.instance() or QApplication(sys.argv)
        screen = app.primaryScreen()
        return screen.size().width(), screen.size().height()

    def _on_mouse_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            self.left_mouse_pressed = pressed
            if self.aim_assist_enabled:
                self.aim_assist_active = pressed
        elif button == mouse.Button.right:  
            self.right_mouse_pressed = pressed
            if self.aim_assist_enabled:
                self.aim_assist_active = pressed

    def _on_key_press(self, key):
        try:
            if key.char == 'v':
                self.aim_assist_enabled = not self.aim_assist_enabled
                self.aim_assist_active = False
                print(f"Aim Assist {'Enabled' if self.aim_assist_enabled else 'Disabled'}")
            elif key.char == '+':
                self.predictor.mouse_sensitivity = min(2.0, self.predictor.mouse_sensitivity + self.sensitivity_step)
                self.overlay.stats['sensitivity'] = self.predictor.mouse_sensitivity
                print(f"Sensitivity increased to: {self.predictor.mouse_sensitivity:.1f}")
            elif key.char == '-':
                self.predictor.mouse_sensitivity = max(0.5, self.predictor.mouse_sensitivity - self.sensitivity_step)
                self.overlay.stats['sensitivity'] = self.predictor.mouse_sensitivity
                print(f"Sensitivity decreased to: {self.predictor.mouse_sensitivity:.1f}")
            elif key.char == '.':
                self.overlay.show_thumbnail = not self.overlay.show_thumbnail
                self.overlay.update()
                print(f"Thumbnail {'shown' if self.overlay.show_thumbnail else 'hidden'}")
        except AttributeError:
            self.predictor.update_aim_offset(key, True)

    def _on_key_release(self, key):
        try:
            if key.char in ['v', '+', '-', '.']:
                return
        except AttributeError:
            self.predictor.update_aim_offset(key, False)

    def _on_key_release(self, key):
        try:
            if key.char in ['v', '+', '-', '.']:
                return
        except AttributeError:
            if key == keyboard.Key.shift:
                self.shift_pressed = False
                if not self.left_mouse_pressed:
                    self.aim_assist_active = False
            else:
                self.predictor.update_aim_offset(key, False)

    def start(self):
        config_path = self._select_config_file()
        if not config_path:
            return False
            
        if not self.detector.load_config(config_path) or not self.detector.init_session():
            return False
            
        self.overlay.classes = self.detector.classes
        self.overlay.show()
        
        try:
            self.camera = dxcam.create(output_idx=0, output_color="BGR")
            self.camera.start(target_fps=144, video_mode=True)
        except Exception as e:
            QMessageBox.critical(None, "Camera Error", f"Failed to start capture:\n{str(e)}")
            return False
        
        self.mouse_listener.start()
        self.key_listener.start()
        
        self._start_main_loop()
        return True

    def _select_config_file(self):
        app = QApplication.instance() or QApplication(sys.argv)
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Select YOLO Config File", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        return file_path

    def _start_main_loop(self):
        last_time = time.perf_counter()
        fps_counter = 0
        fps = 0
        last_fps_update = time.perf_counter()

        def update():
            nonlocal last_time, fps_counter, fps, last_fps_update
            try:
                frame = self.camera.get_latest_frame()
                if frame is None:
                    return

                current_time = time.perf_counter()
                delta_time = current_time - last_time
                last_time = current_time

                fps_counter += 1
                if current_time - last_fps_update >= 1.0:
                    fps = fps_counter / (current_time - last_fps_update)
                    fps_counter = 0
                    last_fps_update = current_time

                infer_start = time.perf_counter()
                detections = self.detector.detect(frame)
                infer_time = time.perf_counter() - infer_start

                tracked_targets = self.predictor.update_targets(detections, current_time)

                if self.aim_assist_enabled and self.aim_assist_active:
                    self._apply_aim_assist(tracked_targets)

                self.overlay.update_overlay(
                    frame=frame,
                    targets=tracked_targets,
                    stats={
                        "fps": fps,
                        "inference": infer_time,
                        "sensitivity": self.predictor.mouse_sensitivity
                    },
                    aim_assist_enabled=self.aim_assist_enabled,
                    aim_assist_active=self.aim_assist_active
                )

            except Exception as e:
                print(f"Main loop error: {str(e)}")
                self.aim_assist_active = False

        self.timer = QTimer()
        self.timer.timeout.connect(update)
        self.timer.start(16)

    def _apply_aim_assist(self, targets):
        if not targets:
            return
            
        mouse_x = self.screen_width // 2
        mouse_y = self.screen_height // 2
        
        closest_target = min(targets, key=lambda t: math.sqrt(
            (t['pred_x'] - mouse_x)**2 + (t['pred_y'] - mouse_y)**2))
        
        dx, dy = self.predictor.calculate_aim_offset(closest_target, mouse_x, mouse_y)
        
        if dx != 0 or dy != 0:
            distance = math.sqrt(dx**2 + dy**2)
            steps = max(1, int(distance / 8))
            
            for _ in range(steps):
                step_x = int(dx/steps)
                step_y = int(dy/steps)
                if step_x != 0 or step_y != 0:
                    self.mouse_controller.move_relative(step_x, step_y)
                    time.sleep(0.001)

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def main():
    if not is_admin():
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        sys.exit()

    if 'DmlExecutionProvider' not in ort.get_available_providers():
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "Error", "DirectML not available.\nPlease install onnxruntime-directml package.")
        return
    
    app = QApplication(sys.argv)
    
    aim_assist = AimAssistSystem()
    if aim_assist.start():
        sys.exit(app.exec_())

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "Error",
            f"Missing required package: {str(e)}\n"
            "Please install with:\n"
            "pip install onnxruntime-directml dxcam pyqt5 opencv-python numpy pyyaml pynput")
        sys.exit(1)