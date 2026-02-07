# Titan Eye Demo

轻量 YOLOv8 + Kalman SORT 跟踪示例，支持视频文件或摄像头输入，并带外观特征匹配、自动阈值微调等防多 ID 处理。

## 环境准备
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行
默认视频源在 `main.py` 顶部的 `video_path`（当前 `testvideo/1.mp4`）。修改后直接运行：
```bash
python3 main.py
```

若改用摄像头，将 `video_path` 设置为设备索引（如 `0`）。窗口中绿色/橙色为跟踪框，ID 稳定时通常为绿色；可在代码中进一步调整过滤与匹配参数。
