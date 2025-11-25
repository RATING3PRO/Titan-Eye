import cv2
from ultralytics import YOLO
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanFilter
from typing import List

# ----------------- 0. 辅助函数和配置 -----------------

def iou_batch(bb_test, bb_gt):
    """
    计算一批测试边界框 bb_test 和一批基准边界框 bb_gt 之间的 IoU。
    """
    if bb_test.size == 0 or bb_gt.size == 0:
        return np.empty((bb_test.shape[0], bb_gt.shape[0]))
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)

def frame_generator(video_path: str):
    """
    逐帧读取器：默认 OpenCV -> FFMPEG -> imageio 三段兜底。
    """
    cap = cv2.VideoCapture(video_path)
    backend = "default"
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        backend = "FFMPEG"
    if cap.isOpened():
        print(f"视频源 '{video_path}' 打开成功，后端: {backend}")
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                break
            yield frame
        cap.release()
        return

    try:
        import imageio.v2 as iio
        print(f"OpenCV 无法打开 '{video_path}'，尝试 imageio 读取...")
        reader = iio.get_reader(video_path)
        for frame in reader:
            yield cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        reader.close()
    except Exception as exc:  # noqa: BLE001
        print(f"错误: 无法读取视频源 '{video_path}': {exc}")


def color_hist_embedding(frame: np.ndarray, box: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    提取 HSV 颜色直方图作为外观特征，返回 L2 归一化的向量。
    """
    x1, y1, x2, y2 = box.astype(int)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(bins * 3, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v], axis=0).flatten()
    norm = np.linalg.norm(hist) + 1e-6
    return (hist / norm).astype(np.float32)


def cosine_sim_matrix(det_embs: List[np.ndarray], trk_embs: List[np.ndarray]) -> np.ndarray:
    if len(det_embs) == 0 or len(trk_embs) == 0:
        return np.empty((len(det_embs), len(trk_embs)))
    det = np.stack(det_embs)  # (M, D)
    trk = np.stack(trk_embs)  # (N, D)
    det_norm = det / (np.linalg.norm(det, axis=1, keepdims=True) + 1e-6)
    trk_norm = trk / (np.linalg.norm(trk, axis=1, keepdims=True) + 1e-6)
    return det_norm @ trk_norm.T  # (M, N)


def nms_detections(dets, iou_thresh=0.5):
    """
    简单的 per-class NMS，输入为字典列表 {box, conf, cls}
    """
    if not dets:
        return []
    keep = []
    by_cls = {}
    for d in dets:
        by_cls.setdefault(d["cls"], []).append(d)
    for _, items in by_cls.items():
        items = sorted(items, key=lambda x: x["conf"], reverse=True)
        suppressed = [False] * len(items)
        for i, det_i in enumerate(items):
            if suppressed[i]:
                continue
            keep.append(det_i)
            box_i = det_i["box"]
            area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
            for j in range(i + 1, len(items)):
                if suppressed[j]:
                    continue
                box_j = items[j]["box"]
                xx1 = max(box_i[0], box_j[0])
                yy1 = max(box_i[1], box_j[1])
                xx2 = min(box_i[2], box_j[2])
                yy2 = min(box_i[3], box_j[3])
                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                union = area_i + area_j - inter
                if union > 0 and inter / union > iou_thresh:
                    suppressed[j] = True
    return keep


def main():
    # ----------------- 1. 初始化和配置 -----------------
    # 指定单个视频源：改这里即可
    video_path = "testvideo/1.mp4"

    # 初始阈值与自适应策略
    yolo_conf = 0.5          # 起始置信度阈值，自适应会上下微调
    min_area_frac = 0.0005   # 最小面积阈值 (占整帧比例)，避免因分辨率变化需要手改
    max_area_frac = 0.8      # 最大面积阈值 (占整帧比例)，避免整屏大框
    classes_str = ""         # 默认全类别，真实场景无需手动修改
    auto_tune = True         # 开启自动阈值调节：无检测时下降阈值，过多检测时提升阈值
    max_dets_per_frame = 30  # 单帧过多检测视为噪声，自动提高阈值
    no_det_frames = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"正在使用的设备: {device}")

    print("正在加载YOLOv8n模型...")
    model = YOLO('yolov8n.pt')  # 使用更快的 'n' 模型
    model.to(device)
    print("模型加载完毕。")

    # 追踪器相关配置
    tracks = []
    next_track_id = 0
    max_age = 25  # 未匹配帧数超过该值则删除轨迹，稍宽容以穿越短时漏检
    iou_threshold = 0.45 # IOU匹配阈值，兼顾重叠不足时的匹配
    duplicate_iou = 0.55  # IoU 达到此阈值直接附着到已有确认轨迹，减少新 ID
    apper_weight = 0.4  # 匹配成本中外观特征的权重（余弦距离），其余为 1-IOU
    
    print("开始处理视频帧 (使用自定义SORT追踪器)...")
    # ----------------- 2. 逐帧处理 -----------------
    frames_seen = 0
    total_dets = 0
    total_conf_tracks = 0
    frame_area = None
    frame_wh = None
    min_area = 0  # 将在读取首帧后基于 min_area_frac 计算
    for frame in frame_generator(video_path):
        frames_seen += 1
        if frame_area is None:
            h, w = frame.shape[:2]
            frame_area = h * w
            frame_wh = (w, h)
            min_area = max(1, frame_area * min_area_frac)

        # --- 预测阶段 ---
        # 预测现有轨迹的新位置
        for track in tracks:
            predicted = track['kf'].predict()
            if predicted is not None:
                track['box'] = predicted
            track['age'] += 1

        # --- 检测阶段 ---
        # classes 解析
        class_list = None
        if classes_str.strip():
            try:
                class_list = [int(c) for c in classes_str.split(",") if c.strip() != ""]
            except ValueError:
                print("警告: classes 解析失败，忽略类别过滤")

        results = model.predict(
            frame,
            conf=yolo_conf,
            classes=class_list,
            verbose=False,
            device=device,
        )
        
        # 获取检测到的边界框
        detections = []
        if results[0].boxes.id is None: # predict模式下id为None
            boxes = results[0].boxes.cpu()
            for box in boxes:
                # 转 numpy，并裁剪到画面内，避免异常大框
                xyxy = box.xyxy[0].cpu().numpy()
                w, h = frame_wh
                xyxy[0] = np.clip(xyxy[0], 0, w - 1)
                xyxy[1] = np.clip(xyxy[1], 0, h - 1)
                xyxy[2] = np.clip(xyxy[2], xyxy[0] + 1, w)
                xyxy[3] = np.clip(xyxy[3], xyxy[1] + 1, h)

                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if min_area > 0 and area < min_area:
                    continue
                if max_area_frac > 0 and frame_area and (area / frame_area) > max_area_frac:
                    continue

                detections.append({
                    'box': xyxy,
                    'conf': float(box.conf[0]),
                    'cls': int(box.cls[0]),
                    'emb': color_hist_embedding(frame, xyxy)
                })

        # NMS
        detections = nms_detections(detections, iou_thresh=0.5)

        # 简单自适应：无检测若连续出现则降低阈值，检测过多则升高阈值
        if auto_tune:
            if len(detections) == 0:
                no_det_frames += 1
                if no_det_frames >= 3 and yolo_conf > 0.25:
                    yolo_conf = max(0.25, yolo_conf - 0.05)
                    no_det_frames = 0
                    print(f"无检测多帧，下调置信度至 {yolo_conf:.2f}")
            else:
                no_det_frames = 0
                if len(detections) > max_dets_per_frame and yolo_conf < 0.7:
                    yolo_conf = min(0.7, yolo_conf + 0.05)
                    print(f"检测过多（{len(detections)}），上调置信度至 {yolo_conf:.2f}")

        # 如果没有检测到目标或没有轨迹，则直接进入下一帧
        if len(detections) == 0 and len(tracks) == 0:
            cv2.imshow("Custom SORT Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # --- 匹配阶段 ---
        # 获取轨迹的预测框
        trk_preds = np.array([t['box'] for t in tracks]) if len(tracks) > 0 else np.empty((0, 4))
        det_boxes = np.array([d['box'] for d in detections]) if len(detections) > 0 else np.empty((0, 4))

        # 计算IOU成本矩阵
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))

        if len(det_boxes) > 0 and len(trk_preds) > 0:
            iou_matrix = iou_batch(det_boxes, trk_preds)
            det_embs = [d['emb'] for d in detections]
            trk_embs = [t.get('emb', np.zeros_like(det_embs[0])) for t in tracks]
            cos_matrix = cosine_sim_matrix(det_embs, trk_embs)

            # 组合成本：外观+位置
            cost_matrix = apper_weight * (1 - cos_matrix) + (1 - apper_weight) * (1 - iou_matrix)
            # 类别不一致时强制高成本，避免人和手机互相匹配导致新 ID
            for di, det in enumerate(detections):
                for ti, trk in enumerate(tracks):
                    if det['cls'] != trk['cls']:
                        cost_matrix[di, ti] = 1.0

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < (1 - iou_threshold):
                    matched_indices.append((r, c))
                    if r in unmatched_detections:
                        unmatched_detections.remove(r)
                    if c in unmatched_tracks:
                        unmatched_tracks.remove(c)

        # --- 更新阶段 ---
        # 更新匹配到的轨迹
        for det_idx, trk_idx in matched_indices:
            track = tracks[trk_idx]
            detection = detections[det_idx]
            track['box'] = track['kf'].update(detection['box'])
            track['age'] = 0
            track['hits'] += 1
            track['cls'] = detection['cls']
            track['conf'] = detection['conf']
            track['emb'] = detection['emb']
            # 只有在连续命中几次后才确认轨迹
            if track['hits'] > 3:
                track['confirmed'] = True


        # 为未匹配的检测创建新轨迹
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            # 与已有确认轨迹重复？如果 IoU 高于 duplicate_iou，直接附给该轨迹并更新，避免新 ID
            attached = False
            for trk_idx, trk in enumerate(tracks):
                if not trk['confirmed']:
                    continue
                box_i = detection['box']
                box_j = trk['box']
                xx1 = max(box_i[0], box_j[0])
                yy1 = max(box_i[1], box_j[1])
                xx2 = min(box_i[2], box_j[2])
                yy2 = min(box_i[3], box_j[3])
                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                union = area_i + area_j - inter
                if union > 0 and inter / union > duplicate_iou and detection['cls'] == trk['cls']:
                    trk['box'] = trk['kf'].update(detection['box'])
                    trk['age'] = 0
                    trk['hits'] += 1
                    trk['cls'] = detection['cls']
                    trk['conf'] = detection['conf']
                    attached = True
                    break

            if attached:
                continue

            kf = KalmanFilter()
            kf.init_state(detection['box'])
            tracks.append({
                'id': next_track_id,
                'kf': kf,
                'age': 0,
                'hits': 1,
                'box': detection['box'],
                'cls': detection['cls'],
                'conf': detection['conf'],
                'emb': detection['emb'],
                'confirmed': False,
            })
            next_track_id += 1

        # 清理过老的轨迹 (在所有操作之后)
        tracks = [t for t in tracks if t['age'] <= max_age]
        
        # --- 绘制结果 ---
        for track in tracks:
            # 只绘制已确认的轨迹
            if not track['confirmed']:
                continue
            
            x1, y1, x2, y2 = map(int, track['box'])
            label = f"ID:{track['id']} {model.names[track['cls']]} {track['conf']:.2f}"
            color = (0, 255, 0) if track['age'] == 0 else (0, 165, 255) # 绿色表示当前帧匹配，橙色表示预测
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        total_dets += len(detections)
        total_conf_tracks += sum(1 for t in tracks if t['confirmed'])

        cv2.imshow("Custom SORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户中断。")
            break
    else:
        if frames_seen == 0:
            print(f"错误: 未能读取到任何帧，请检查视频文件或解码依赖。")
        else:
            print(f"视频处理完毕。帧数: {frames_seen}，检测数: {total_dets}，确认轨迹数累积: {total_conf_tracks}")
            
    # ----------------- 4. 释放资源 -----------------
    cv2.destroyAllWindows()
    print("资源已释放。")

if __name__ == "__main__":
    main()
