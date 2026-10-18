# SAM3 Video Service v4 — 流式输入及启动预加载加固版

## 1. 审计与架构结论

来源：用户上传的 `sam3-service-manage-master.zip`，重点文件：
- `sam3_video_service.py`：请求阶段执行 `get_predictor()`，每个请求中可能冷加载模型；秒级临时文件名；异常时资源不保证释放；整文件输入。
- `sam3_video_service_fixed.py`：已增加 FFmpeg 截帧和部分分割响应修复，仍在请求内惰性调用预测器，独立 `/video/stream` 不存在。
- `scripts/start_sam3_video.sh`：默认从旧 Windows 挂载工作区加载另一份源码，容易误部署；仅验证端口存活。
- `scripts/manage_all_mvp_apis.sh`：使用宽泛 `pkill -f` 模式，不适合生产进程管理。

**关键接口约束：**
1. Meta 原生 SAM3.1 `handle_request` 不提供通用的 `add_frame`，已有的
   `propagate_in_video` 是对先前加载的视频传播。单纯新增 WebSocket 无法令它支持持续输入。
2. Transformers `Sam3VideoModel` + `Sam3VideoProcessor` 支持
   `init_video_session(video=None)`、`add_text_prompt()` 和逐帧 `model(session, frame=...)`。
3. HF 流式模型与当前 `.pt` 的 Meta 原生 SAM3.1 权重格式、运行路径不同，需要单独安装依赖及验证权重和准确率。

因此提供**两种明确的单模型启动模式**：
- `SAM3_ENGINE=meta`：保持原本离线分割推理，`/video/stream` 明确返回 `UNSUPPORTED_BACKEND`。
- `SAM3_ENGINE=hf`：使用 Transformers 兼容 SAM3 视频权重，启用真正逐帧 WebSocket 推理；
  整文件端点也可使用同一在线后端逐帧解码。上线前独立重新计算 IoU。

**切勿同时在同一 RTX 3090 上运行新旧两套大模型。**

## 2. 实现功能

- FastAPI `lifespan` 任务在服务启动时开始加载模型。`/health/live` 在进程存活时为200；
  `/health/ready` 仅模型真正加载成功才为200，加载中/失败/显存不足为503。
  HTTP推理请求从不发起模型惰性加载。
- 异步接口将阻塞的 GPU 推理交给线程执行，HTTP 和 WebSocket 共用一个
  推理许可及进程内线程锁，防止请求取消后 GPU 线程仍执行时出现重入。
- 接口：`GET /health/live`、`GET /health/ready`、`GET /health`、
  `GET /v1/capabilities`、`POST /video/segment`、
  `WS /video/stream`（另有 `/v1/` 同义路径）。
- HTTP 大文件分块落盘，按 SHA256 标识；FFprobe 独立检查总帧数和尺寸；
  FFmpeg 在模型打开之前**真实生成限制帧数的短视频**；原始全长无采样时直接使用上传路径。
- 源视频帧序号和抽样后帧序号分别返回；检查遗漏、重复及对象掩码结构。
- WebSocket 二进制 JPEG/PNG 帧按ACK串行发送，无无限增长的应用帧队列。
  最大帧大小、像素、总帧数、会话持续时间、空闲超时、输出大小均可配置。
  自动关闭后端状态并释放推理许可。
- API Key 采用安全比较；未设置密钥时仅允许回环客户端；
  拒绝跨站浏览器 Origin；`run.py` 在免认证模式下强制绑定127.0.0.1。
- 启动/停止脚本只追踪由自己启动的PID，检测端口占用，**不会执行宽泛pkill**。
- 捕获可恢复的显存不足时进入`degraded`，就绪检查失败，避免持续OOM重试。

### 资源限制默认值

为避开之前 1545 帧输入导致的 OOM，默认限制为400帧及640×480；
按实际机器性能测试后，可以显式增大 `SAM3_MAX_FRAMES` 和
`SAM3_MAX_FRAME_PIXELS`。限制仅约束已知输入大小；RAM、显存、进程外OOM仍须监控。

## 3. 安装（不覆盖现有服务）

假设 ZIP 解压后目录为 `/opt/molink/app/sam3-service-v4`：

```bash
cd /opt/molink/app/sam3-service-v4
/opt/molink/app/.venv/bin/python3 -m pytest -q sam3_service_v4/test_v4.py
```

基础依赖：你现有的 PyTorch/SAM3、FastAPI、Uvicorn、python-multipart、
Pillow、NumPy、OpenCV、FFmpeg/FFprobe，以及运行测试所需的 Pytest。
请先核对当前虚拟环境的已安装版本，避免为了部署破坏经过验证的SAM3.1依赖。

### A. 保持现有已通过IoU的 Meta SAM3.1（离线模式）

在旧服务彻底停止、GPU1空闲且端口6014已释放后：

```bash
cd /opt/molink/app/sam3-service-v4
export CUDA_VISIBLE_DEVICES=1
export SAM3_DEVICE=cuda:0
export SAM3_ENGINE=meta
export ALLOW_UNAUTH_LOCAL=1
export SAM3_MODEL_PATH=/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/sam3.pt
export SAM3_BPE_PATH=/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/bpe_simple_vocab_16e6.txt.gz
export SAM3_MAX_FRAMES=400
export SAM3_MAX_FRAME_PIXELS=307200
export PYTHONPATH="$PWD"
./scripts/start_sam3_v4.sh
```

检查：

```bash
curl -fsS http://127.0.0.1:6014/health/ready
curl -fsS http://127.0.0.1:6014/v1/capabilities
```

若环境变量中未设置密钥，HTTP免认证仅支持本机回环；外部访问需要 `API_KEY`
并配置可信的 `SAM3_BIND_HOST`。`CUDA_VISIBLE_DEVICES=1` 下进程内显卡为 `cuda:0`。

原始400帧回归测试（整文件上传）：

```bash
curl -fSs --max-time 900 -X POST http://127.0.0.1:6014/video/segment \
  -F 'file=@/mnt/c/Users/root/Downloads/sam3_calligraphy_brush_640x480_20fps_20s.mp4' \
  -F 'text_prompt=brush' -F 'frame_stride=1' -F 'max_frames=400' \
  -F 'return_masks_png=true' -o /tmp/sam3_v4_brush_masks.zip

/opt/molink/app/.venv/bin/python3 scripts/evaluate_sam3_iou.py \
  --pred /tmp/sam3_v4_brush_masks.zip \
  --gt /mnt/c/Users/root/Downloads/sam3_brush_ground_truth.zip \
  --target brush --outdir /tmp/sam3_v4_eval
```

精度验收：平均IoU≥0.90且至少360/400帧IoU≥0.90；新版本需要重新测试，
历史93.87%仅作为基准。

### B. 真正逐帧输入（HF流式模式）

准备**Hugging Face Transformers 可加载的 SAM3 Video 模型目录**（包括config、
处理器/分词器文件以及兼容的权重）。`SAM3_MODEL_PATH` 的原生SAM3.1 .pt
不等同于此目录。

先在单独测试虚拟环境中验证（防止破坏Meta原生依赖）：

```bash
python -c 'from transformers import Sam3VideoModel,Sam3VideoProcessor; print("HF Sam3Video available")'
```

然后在旧进程/GPU已释放的前提下：

```bash
export SAM3_ENGINE=hf
export HF_SAM3_MODEL=/path/to/transformers-compatible-sam3
export CUDA_VISIBLE_DEVICES=1
export SAM3_DEVICE=cuda:0
export HF_SAM3_DTYPE=float32
export ALLOW_UNAUTH_LOCAL=1
export PYTHONPATH=/opt/molink/app/sam3-service-v4
python -m sam3_service_v4.run
```

以第二终端真实逐帧上传前10帧，服务器处理完一帧才发送下一帧：

```bash
python -m sam3_service_v4.stream_client \
  --video /mnt/c/Users/root/Downloads/sam3_calligraphy_brush_640x480_20fps_20s.mp4 \
  --prompt brush --frames 10 \
  --output /tmp/sam3_hf_10frames.zip
```

流式质量可能低于全视频处理：部分依赖未来帧的目标确认策略不可用。
运行400帧时需要另行验证连续跟踪、每帧IoU、CPU/GPU内存增长和吞吐率。

## 4. WebSocket协议

连接：`ws://127.0.0.1:6014/video/stream`，若设置API_KEY，使用
`Authorization: Bearer ...` 请求头。客户端先发送JSON：

```json
{"type":"start","prompt":"brush","max_frames":10,"return_masks_png":true,"output_prob_thresh":0.1}
```

服务器返回：

```json
{"type":"ready","session_id":"...","ack_required":true,"next_frame_index":0}
```

每帧可先发送可选元数据，然后发送PNG/JPEG **二进制**帧：

```json
{"type":"frame_meta","frame_index":0,"timestamp_ms":0}
```

服务器完成推理后才返回：

```json
{"type":"frame_result","frame_index":0,"num_objects":1,
 "objects":[{"object_id":3,"score":0.81,"mask_pixels":1234,"mask_png_base64":"..."}],
 "inference_ms":42.0,"next_frame_index":1}
```

每帧必须先收到ACK再发送下一帧，以实现背压。达到帧数上限返回`type=done`。
可发送`{"type":"end"}`或`{"type":"cancel"}`结束并清理会话。
数据损坏、超限和顺序错误均返回明确错误并关闭会话。

## 5. 当前验证范围与遗留项

本地：
- **25项无GPU单元/ASGI集成测试通过**。
- **独立进程的真实TCP/WebSocket测试通过**：客户端传入3帧PNG，
  服务器逐帧返回ACK与掩码（模拟后端）。
- 测试也覆盖10帧物理视频截断、身份认证、跨站限制、异常清理、
  HF接口适配器结构和旧版Meta会话表兼容。

尚未完成：
- 在你的Molink-Infer运行真正的SAM3.1权重和新代码。
- 在独立HF环境加载真正的兼容权重，验证真实流式预测。
- 在新版本复测400帧IoU，以及验证长期内存曲线和实时吞吐率。
- 超长视频系统级OOM无法仅靠Python捕获；生产部署可额外施加cgroup/WSL内存限制并监控。
- 不能把流式后端的精度自动视为已通过的Meta原生离线精度。

## 6. 重要回滚措施

不要在确认前直接覆盖当前 `/opt/molink/app/sam3_video_service_fixed.py`
或工作区旧服务。先备份旧进程启动命令、源码SHA256、权重路径和
`sam3-masks2.zip`等已通过IoU的产物。切换失败可停止v4进程，再恢复原启动命令；
不要用 `pkill -f uvicorn` 杀死其他MVP服务。

官方文档参考：
- Meta视频预测器：https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_base_predictor.py
- Transformers SAM3 Video流式说明：https://huggingface.co/docs/transformers/model_doc/sam3_video
