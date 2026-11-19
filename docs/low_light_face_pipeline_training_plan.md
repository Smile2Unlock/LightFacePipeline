# 弱光全链路人脸认证模型训练规划（检测 + 5点对齐 + 识别 + 时序活体）

**更新日期：** 2026-11-19  
**范围限定：** 纯模型训练、微调、知识蒸馏、ONNX 导出与离线端到端验收；**不包含任何主软件代码接入与工程集成**。  
**三大工程前提：**
1. **非商用 / 个人研究用途**：可合法使用学术研究级预训练权重与公开数据集（如 DarkFace、WebFace4M/12M 预训练权重、CelebA-Spoof、OULU-NPU、SiW 等），但仍须为自留归档锁定数据源、commit 与 SHA-256。
2. **64GB 显存 ROCm 兼容计算卡**：全流程严格限定在**纯 PyTorch 原生算子 + ROCm 稳定生态**，彻底规避强绑定 NVIDIA CUDA C++/PTX 扩展的依赖。
3. **原生时序活体**：活体阶段直接训练基于连续视频片段的轻量化时空模型，替代“单帧拼外置滑窗”的弱时序路线。

---

## 与本项目已有基线的对齐

本训练规划继承并升级了 [lightweight_face_detector_plan.md](file:///home/ation_ciger/Projects/Smile2Unlock_v2/docs/lightweight_face_detector_plan.md) 与现有研究记录中的关键实测边界：

- **体量瓶颈**：现有 SeetaFace 模型总计约 **162.95 MiB**，其中检测模型仅占约 `3.87 MiB`、landmarker 约 `0.40 MiB`，而识别与两份反欺骗模型占约 `158.7 MiB`。因此必须“检测 + 识别 + 活体”一条龙重构，才能将总模型体积压缩到 **~14~24 MB（FP32 ONNX）**（FP16/INT8 后 **~5~12 MB**，瘦身 **85%~95%**）。
- **DarkFace 边界**：本地 `build/research/darkface/` 已有 `6,000` 张标注图像（`50,399` 个人脸框，平均每图 `8.40` 张脸），其中 `82.2%` 人脸高度 `<24px`、`92.0%` `<34px`。现有 SeetaFace `@640` 在 6,000 张上的评测基线仅为 AP50 `0.145%`（当前 gamma 重试 `app640` 为 `0.614%`，见 [build/research/darkface/README.md](file:///home/ation_ciger/Projects/Smile2Unlock_v2/build/research/darkface/README.md)）。同时 DarkFace 官方包还包含 **`9,000` 张无标注夜间暗图**与 **`789` 对真实低/正常光照配对图像**——以前仅作 6,000 张基线测试，本方案将它们全部激活用于半监督与配对训练，但**绝不将无关键点/ID/活体标签的 DarkFace 误用于识别或活体监督**。
- **真实传感器与逆光条件**：
  - [backlight_exposure.md](file:///home/ation_ciger/Projects/Smile2Unlock_v2/docs/backlight_exposure.md) 记录过真实窗前逆光场景：人脸均值亮度低至 `12/255`、背景窗高达 `193/255`，且硬件手动曝光上限为 `1/30s`（对应明显的高 ISO 噪点与轻微运动模糊）。
  - [windows-camera-liveness.md](file:///home/ation_ciger/Projects/Smile2Unlock_v2/docs/windows-camera-liveness.md) 记录了真实 UVC 采集链路为 `MJPEG / YUY2 / NV12`（15+ FPS，≤1280×720），并揭示过历史 RGB/BGR 通道反转导致匹配与活体漂移的隐患。因此本方案在训练期显式模拟 YUV 色度下采样/JPEG 块效应，并全程锁死 **RGB** 契约。

---

## 一、技术先进性对比与总体架构（2025–2026 SOTA 升级清单）

相对 2021–2023 年常见的“微调 YuNet + ArcFace/IR101 蒸馏 MobileFaceNet + 单帧 MiniFASNet/老式 TSM”路线，本规划全面升级为 **2025–2026 最新验证有效、且兼容 ROCm 与轻量 CPU 推理的 SOTA 技术栈**：

| 模块 | 传统 2021–2023 方案 | 本方案 2025–2026 SOTA 升级路线 | 核心收益与依据 |
|---|---|---|---|
| **1. 检测 + 5点** | 静态 gamma LUT + 普通微调 YuNet-N（仅用 6,000 有标注图） | **任务驱动可学习低照度曲线前端（LCF，Zero-DCE 思想，<30K 参数）+ `YOLO11n-face-5pt`（纯 PyTorch Anchor-Free + TAL + DFL + WingLoss）**；激活 DarkFace **789 真实亮暗配对 + 9,000 无标注暗图 EMA Mean-Teacher** + HLA 风格跨光照特征蒸馏（保留 `YuNet-S + LCF` 作 `<1MB` 极轻备选） | 1. 静态 LUT 会在极暗区暴增纯噪点；可学习 LCF 按检测与 5 点梯度自动抑制暗噪并拉出五官边界（新增延迟 `<0.2ms`）。<br>2. 彻底摆脱 `mmdet/mmcv` 编译地狱，100% 适配 ROCm。<br>3. 利用 9,000 张无标注真实夜间传感暗图补齐合成退化盲区。 |
| **2. 跨光照识别** | ResNet100-ArcFace 离线蒸馏 MobileFaceNet / IResNet18 | **CVLface `ViT-Base-KP-RPE`（CVPR 2024+，WebFace12M 预训练）作冻结 Teacher + `AdaDistill` 课程动态类中心蒸馏 + `AdaFace` 质量自适应 Margin** → 蒸馏至 **`EdgeFace-S / EdgeFace-XS`（IEEE T-BIOM 2024 / IJCB 高效竞赛冠军，1.77M~3.65M 参数）** | 1. **KP-RPE** 将 5 点几何偏置注入 ViT 注意力，对暗光 5 点微抖动鲁棒性显著强于普通 CNN Teacher。<br>2. **EdgeFace**（SDTA 轻量注意力 + LoRaLin 低秩线性层）在低画质下以 1.77M~3.65M 参数远超传统 MobileFaceNet。<br>3. **AdaFace + AdaDistill** 用特征模长自适应放松暗噪点样本 Margin，并把暗图学生输出精准锚定在亮图 Teacher 超球面上。 |
| **3. 时序活体** | CLIP/ResNet 单帧活体 + 外置规则均值，或老旧 TSM | **双流大模型 Teacher（`DINOv2 with Registers` ViT-B/14 + LoRA 全局流，ICCVW 2025 `arXiv:2510.17201` / AAAI 2026 + ConvNeXtV2-Laplacian 高频纹理流）** → 蒸馏至 **双尺度质量感知时序活体 `ST-FAS-Lite`（`ConvNeXtV2-Atto` / `MobileNetV4-Conv-S` + 帧差/CDC 纹理茎 + 1D-DWConv & 2层轻量 Temporal Transformer + `FIQA-FAS` 帧级质量加权池化）**；叠加 **帧冻结+刚性2D手抖硬负样本（时序伪造解耦思想）+ 伪 3D 深度图头 + 暗室自发光差异化物理增广** | 1. **DINOv2-with-Registers** 的寄存器 token 消除了常规 ViT 的背景高范数伪影，跨设备/跨光照域泛化远超原生 CLIP。<br>2. **FIQA-FAS** 在时序池化中自动降权单帧运动模糊/极黑废帧。<br>3. **真视频帧冻结+刚性2D手抖**从根源封死“仅靠均值亮度/静态清晰度作弊”和“2D照片微晃绕过时序模型”的漏洞。 |

### 全链路产出模型体量与结构总览

```text
原始 RGB 帧流 (t-7 ... t)
  │
  ├──▶ [Model 1] LCF + YOLO11n-face-5pt (~5~6 MB FP32; 备选 YuNet-S+LCF ~0.8 MB)
  │       └─ 输出: bbox(x1,y1,x2,y2), det_score, 5-pts (RGB 图像绝对坐标)
  │
  ├──▶ (5-pt Umeyama 112x112 仿射对齐) ──▶ [Model 2] EdgeFace-S / XS (~7~14 MB FP32)
  │                                           └─ 输出: 512-d L2 特征 + 特征模长质量指示 ||z||
  │
  └──▶ (按最近 T=8 帧 bbox 裁剪双尺度 1.5x & 2.7x 各 128x128 RGB)
          └──▶ [Model 3] ST-FAS-Lite (~3~5 MB FP32, 支持推理期逐帧空间特征缓存)
                  └─ 输出: live_prob, spoof_type(real/print/replay/frozen), clarity_score
```

**整套三模型合计体积**：FP32 ONNX 约 **`15 ~ 25 MB`**（若识别选 `EdgeFace-XS` 且活体用 `MobileNetV4-Conv-S`，可压至 **`11 ~ 15 MB`**；FP16/INT8 后约 **`5 ~ 12 MB`**）。

---

## 二、全链路统一模型契约（跨模块对接标准）

即使本阶段不接入软件，训练前也必须把三个模型的 I/O 边界完全锁死，确保离线串联与未来任意集成方零摩擦对齐。

### 1. 像素格式与归一化
- **颜色空间**：全链路统一使用 **RGB (`uint8` 转 `float32` 除以 `255.0`)**，严禁部分模型吃 BGR、部分吃 RGB。
- **检测器输入**：动态多尺度或标准 `640×480` / `640×640`（等比缩放 + `114/255` 灰边 Letterbox，或直接保持 32 倍数长宽），模型内部自带 `LCF` 曲线增强前端，**输入必须是未经人工 gamma 拉亮的原始 RGB 帧**。
- **识别器输入**：`[1, 3, 112, 112]` RGB，归一化 `(x / 255.0 - 0.5) / 0.5`（即 `[-1, 1]`）。
- **时序活体输入**：`[1, T, 6, 128, 128]`（或拆分为 `1.5×` 纹理流 `[1, T, 3, 128, 128]` 与 `2.7×` 上下文流 `[1, T, 3, 128, 128]`，**推荐使用固定 ImageNet 均值方差归一化，或保持 `[0, 1]` 后在模型内做固定 BN/Affine**）。**推理阶段提供两种等价导出**：
  - 整段 ONNX：`frame_clip_tex [1, 8, 3, 128, 128]` + `frame_clip_ctx [1, 8, 3, 128, 128]`（用于离线评测）。
  - 流式双 ONNX（推荐未来线上使用，本阶段须验证两者数值误差 `< 1e-4`）：
    1. `fas_spatial_enc.onnx`：单帧编码器，吃当前帧双尺度图 + 上一帧单帧缓存图（用于内部差分），输出当前帧 `spatial_feat [1, D]` 与 `frame_quality [1, 1]`；
    2. `fas_temporal_head.onnx`：轻量时序头，吃最近 `T=8` 帧缓存队列 `[1, 8, D]` 与 `[1, 8, 1]`，输出最终 `live_prob`、`spoof_logits[4]` 与综合 `clarity_score`。

### 2. 5 点关键点标准与 112×112 对齐模板
- **5 点坐标顺序（严格对应图像视角左右）**：
  `[0] 左眼中心 (图像偏左)`, `[1] 右眼中心 (图像偏右)`, `[2] 鼻尖`, `[3] 左嘴角`, `[4] 右嘴角`
- **标准 ArcFace / CVLface 112×112 Umeyama 仿射目标模板**（单位：像素）：
  ```python
  ARCFACE_DST_112 = np.array([
      [38.2946, 51.6963],
      [73.5318, 51.5014],
      [56.0252, 71.7366],
      [41.5493, 92.3655],
      [70.7299, 92.2041],
  ], dtype=np.float32)
  ```
  > 切记：每次做推理对齐都使用同一套相似变换（Umeyama `estimate_affine_partial_2d`）且从**未增强、未经暗光 gamma 拉伸的原始视频帧**裁剪。

### 3. 时序活体片段采样契约
- **片段长度与跨度**：默认 **`T = 8` 帧**，帧间物理间隔 **`80ms ~ 125ms`**（15fps 相机取隔帧，30fps 相机每 3 帧取 1 帧），覆盖约 **`0.6s ~ 1.0s`** 真实微动作、屏幕刷新/频闪与自动曝光微波动周期。
- **双尺度空间裁剪（均从原图坐标系直接裁剪，不从 112×112 放大）**：
  - 首先由检测框 `bbox(x1, y1, x2, y2)` 计算正方形参考中心 `(cx, cy)` 与边长 `S = max(w, h)`；
  - **纹理分支 (`1.5×`)**：以 `(cx, cy)` 为中心、边长 `1.5 * S` 裁剪并双线性缩放到 `128×128`（越界补边缘零值或均值灰）；
  - **上下文分支 (`2.7×`)**：以 `(cx, cy)` 为中心、边长 `2.7 * S` 裁剪并双线性缩放到 `128×128`（专用于捕获手机屏幕边框、光晕、纸张边缘与肩颈-背景光照连续性）。

---

## 三、64GB 显存 ROCm 兼容卡工程与避坑指南

因为服务器采用的是**非 NVIDIA 但支持 ROCm 的 64GB 显存卡**，软件栈选型直接决定是“高效跑通”还是“卡死在编译和算子报错上”。

### 1. 绝对依赖黑名单与纯 PyTorch 替代栈

| 严禁引入（高风险/CUDA 绑定） | 问题说明 | 本方案指定替代（100% ROCm 安全） |
|---|---|---|
| `mmcv` / `mmcv-full` / `mmdet` | 大量自定义 CUDA C++ 算子（DeformConv/RoIAlign/NMS 编译极易在非主流 ROCm 架构挂掉） | **纯 PyTorch 原生实现 + `torchvision.ops.nms`**；检测基于 Ultralytics 纯 PyTorch 架构（或纯 PyTorch 重写版 YuNet） |
| `flash-attn` (CUDA wheel) / `xformers` | 编译强依赖特定 NVIDIA Compute Capability 或特定 AMD gfx 白名单 | **PyTorch 原生 `F.scaled_dot_product_attention` (SDPA)** |
| NVIDIA `apex` / `DALI` / `bitsandbytes` | 仅限 NVIDIA 生态或需要复杂 ROCm fork | **`torch.amp.autocast` + PyTorch 矢量 GPU 物理退化模块 + `timm` / `transformers` / `peft`** |
| `torch.compile(..., mode="max-autotune")` | 早期或非主流 ROCm 卡上 Triton 内核可能报错或编译卡死 | **训练默认关闭 `torch.compile`**；仅在 Phase 0 验证稳定后才考虑开启 |

### 2. Phase 0 第一步：5 分钟 ROCm 环境压测清单
正式开训前必须在目标服务器上跑一段最小基准脚本，确认以下 4 项：
1. **精度模式判定（关键！）**：测试 `torch.amp.autocast('cuda', dtype=torch.bfloat16)` vs `dtype=torch.float16 + GradScaler` 在 Conv2d + Attention 上的步耗时与 Loss 稳定性。
   - 若该卡原生支持硬件 `BF16` 矩阵加速：全流程优先使用 **`BF16`**（无需 `GradScaler`，抗数值溢出能力碾压 `FP16`，极适合 Partial FC 与 ViT 蒸馏）；
   - 若该卡 `BF16` 走软件回退极慢，则统一改用 **`FP16 + GradScaler`**，并在 `AdaFace` / `Cosine` 相似度计算处显式 `.float()` 保持 FP32。
2. **SDPA 后端探测**：用 `torch.nn.attention.sdpa_kernel` 验证 `FLASH_ATTENTION` / `EFFICIENT_ATTENTION` / `MATH` 哪种后端在该卡不报 kernel error 并吞吐最高（最差情况回退至 `MATH` 在 ViT-B 仅 256 个 patch token 下依然很快）。
3. **显存碎片管理**：设置 `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`（或对应 ROCm 版本环境变量），避免动态多尺度检测训练引发显存碎片 OOM。
4. **多进程 DataLoader 禁忌**：禁止在 `DataLoader` 的子进程中调用 GPU Tensor 做数据增广（会触发 HIP 上下文重复初始化崩溃）。正确模式：**CPU 子进程仅做解码/基础裁剪并输出 `uint8` Tensor，`pin_memory=True` 传入主 GPU 后，在 GPU 上执行物理弱光退化**。

### 3. 64GB 单卡 Teacher + Student 同驻显存预算分配

64GB 显存最大的红利就是：**无需先费硬盘写死 Teacher 离线特征（否则无法做在线强随机弱光/仿射抖动增广），可以直接把冻结大模型和轻量学生同驻显存做“真·在线孪生视图蒸馏”**：

| 阶段 | 同驻模型组件 | 建议 Batch Size | 预估峰值显存占用 | 余量用途 |
|---|---|---|---|---|
| **Phase 1 检测** | 冻结 Teacher (`RetinaFace-R50` 或 `YOLO11l`) + Student (`YOLO11n-5pt + LCF`) + Mean-Teacher EMA | `128 ~ 256` (多尺度最大 `640²`) | `~28 ~ 42 GB` | GPU 在线退化缓冲 + EMA 副本 |
| **Phase 2 识别** | 冻结 Teacher (`CVLface ViT-B-KP-RPE`) + Student (`EdgeFace-S`) + Partial FC (10%~20% 中心) | `512 对` = `1024 张 112² 图/step` | `~22 ~ 34 GB` | 双视图对比 + 稳定 BN 统计 |
| **Phase 3 活体** | 冻结 Teacher (`DINOv2-Reg ViT-B/14` + `ConvNeXtV2-T`, 阶段二仅前向不存梯度) + Student (`ST-FAS-Lite` 8帧) | `64 ~ 128 个 8 帧视频片段` (`512 ~ 1024 帧/step`) | `~32 ~ 48 GB` | 双尺度 `1.5×` & `2.7×` 显存批处理 |

---

## 四、Phase 0：物理级弱光/逆光退化引擎与真机弱光验证集

### 1. 纯 PyTorch GPU 矢量物理退化管线（5 步闭环）

严格禁止简单用 `img * 0.2` 或普通 `ColorJitter` 替代。设计一个纯 PyTorch GPU 模块 `PhysicalLowLightDegrader(nn.Module)`，支持对整批 `[B, T, C, H, W]` Tensor 在 GPU 上高速在线执行以下物理链条：

1. **Inverse ISP 线性光域曝光衰减**：
   - 将 sRGB 归一化图 `[0, 1]` 反解回线性辐照度域：$L = I^{\gamma_{\text{inv}}}$，其中 $\gamma_{\text{inv}} \in [2.0, 2.4]$ 随机采样；
   - 乘以全局暗光曝光系数 $\alpha_{\text{exp}} \in [0.04, 0.50]$（可覆盖极暗与轻暗）。
2. **空间非均匀光照场调制（专治 Smile2Unlock 真实场景）**：
   以概率随机组合以下 3 种空间权重场 $M(x, y)$ 乘在人脸与背景区域：
   - **强窗前逆光（Backlight，核心场景）**：利用 GT 人脸框生成柔和椭圆/高斯 Mask，让人脸区域照度衰减至均值 luma `12~55/255` 区间，同时把非人脸背景拉亮至 `160~255/255`（局部高光允许轻微溢出截断），直接复现 [backlight_exposure.md](file:///home/ation_ciger/Projects/Smile2Unlock_v2/docs/backlight_exposure.md) 中的实际场景；
   - **屏幕冷光夜间补光**：从画面下偏中心位置施加径向冷白光衰减（模拟关灯后仅靠笔记本/显示器屏幕照亮正脸，背景全黑）；
   - **单侧阴影 / 顶光帽檐阴影**：在人脸区域施加线性或斜向阴影梯度，制造半脸亮、半脸暗。
3. **线性域真实传感器噪声 + 色温色偏**：
   - **散粒噪声（Shot Noise，信号相关）**：在线性域按泊松近似高斯 $n_{\text{shot}} \sim \mathcal{N}(0, \sigma_s^2 \cdot L)$ 采样，$\sigma_s \in [0.005, 0.04]$；
   - **读出噪声（Read Noise，含暗电流与轻微通道不均）**：$n_{\text{read}} \sim \mathcal{N}(0, \sigma_r^2)$，$\sigma_r \in [0.003, 0.025]$，并对 `R/B` 通道额外叠加轻微增益漂移模拟暗光自动白平衡（AWB）偏暖黄（2800K）或偏冷青（7500K）；
   - 正 Gamma 映射回 sRGB：$I_{\text{srgb}} = \text{clamp}(L_{\text{noisy}}, 0, 1)^{1/\gamma_{\text{inv}}}$。
4. **光学曝光模糊（模拟 `1/15s ~ 1/30s` 曝光限制）**：
   - 随机 30%~50% 概率施加轻微方向性运动模糊或高斯失焦模糊（卷积核 `3×3` ~ `7×7`）。
5. **UVC 传输色度下采样与压缩伪影（还原摄像头真实画质）**：
   - 先做 `RGB → YCbCr`，按真实 `NV12/YUY2` 将 `Cb/Cr` 下采样 `2×` 再双线性升采样回原尺寸，转回 `RGB`；
   - 再以可微/近似块效应方式叠加 `8×8` DCT 量化网格噪声或离线随机 `JPEG quality ∈ [45, 85]` 压缩（还原 `MJPEG` 暗光色块涂抹）。

### 2. 时序视频级一致性红线
当该引擎处理 **Phase 3 的 `T=8` 帧视频片段** 时，必须遵守：
- **光照场、色温偏移、曝光系数 $\alpha_{\text{exp}}$**：在 8 帧内部保持一致或加入连续缓慢微漂移（模拟自动曝光或头微动引起的亮度微变）；
- **传感器散粒/读出噪声、压缩网格抖动**：**必须每帧独立重采样（i.i.d.）**！绝对不能整段 8 帧贴同一张固定噪声纹理，否则时序活体模型会把“每帧都在闪烁的高频噪点”和“静止不动”学反。

### 3. 三模块亮度下限安全边界（严防拟合纯噪点垃圾）
- **检测器**：下探极暗区，人脸框内原图平均亮度可低至 **`10~15 / 255`**（只要五官轮廓仍可辨，目的是能触发定位与提示）。
- **识别器**：人脸区均值亮度严格控制在 **`25 ~ 220 / 255`**。如果随机衰减后人脸区域均值 `<22/255`，自动抬升 $\alpha_{\text{exp}}$ 回调到安全下限，**绝不强迫模型去记住全黑纯噪点图的身份 ID**。
- **活体模型**：**严禁真假样本使用同一套均匀压暗参数**（详见 Phase 3 差异化物理增广）。

### 4. 自采“真机弱光端到端验证集”（建议第 1 天先完成，全程绝对不入训练集）
找 **3~5 名志愿者**、使用 **2~3 台普通设备**（如笔记本自带 720p 摄像头 + 普通 USB UVC 摄像头），保持真实桌面距离，按 **4 档光照** 录制 3~5 秒短视频（15~30fps 原始未调色帧）：
1. **正常室内光**（基线对照）
2. **强窗前逆光**（背对亮窗/台灯直射背景，人脸明显黑）
3. **极暗台灯 / 环境微光**
4. **关灯仅显示器冷光补光**

覆盖 **3 类业务切片**：
- **检测切片**（~300~500 抽帧）：包含单人正脸、大角度偏头、戴眼镜反光、**多人同框**、**无人脸暗室空帧**，手工标框与 5 点。
- **识别切片**：每人正常光注册 3~5 帧，其余 3 种弱光/逆光/戴眼镜视频抽帧作为探针（形成真实跨光照 1:1 与 1:N 对照）。
- **活体切片**：在全部 4 档光照下录制 **真人** + **手机/平板翻拍（含 20% 暗屏、50% 正常屏、100% 亮屏各播照片与视频）** + **A4 彩色/黑白打印纸（静态与微弯晃动）**。至少单独留出 **1~2 人的全部视频作完全未见人跨设备验证集**。

---

## 五、Phase 1：弱光人脸检测 + 5 点关键点模型训练

### 1. 模型架构选型

#### 主选方案：`LCF + YOLO11n-face-5pt`（~5~6 MB FP32 ONNX，ROCm 零痛点）
1. **前置可学习低照度曲线前端（LCF，Learnable Curve Frontend）**：
   - 借鉴 Zero-DCE 的任务驱动思想，但极度轻量化（**3 层 depthwise-separable Conv，`C=16`，参数量 `<30K`**）：输入原图 RGB（下采样到 `1/4` 分辨率预测低频像素级曲线图 $A(x,y) \in [-1, 1]^3$ 后双线性上采样回原分辨率，经 4 次轻量化迭代曲线 $I_{k+1} = I_k + A \cdot I_k(1 - I_k)$ 自适应提亮暗脸、同时保护亮窗不过曝）。
   - **关键点**：**没有任何独立图像增强 Loss（不做 PS 画质重建）**，LCF 的全部梯度直接来自下游 **检测框 + 5 点关键点损失**，保证它只强化有利于找框与精确定位五官的局部边缘。
   - **轻量化旁路**：当全局输入平均亮度已经足够亮且非逆光时，模型内自然学出接近零的残差 $A \approx 0$，保证正常光不损坏纹理。
2. **主干与检测头（纯 PyTorch Anchor-Free）**：
   - 基于 Ultralytics `YOLO11n-pose`（改 5 点关键点输出，`kpt_shape=[5, 3]` 含可见性/置信度）或直接采用纯 PyTorch 实现的 P3/P4/P5（stride 8/16/32）Anchor-Free 检测头 + Task-Aligned Assigner (TAL) + DFL 框回归 + **WingLoss 5 点回归**。
   - 对接 [build/research/darkface/README.md](file:///home/ation_ciger/Projects/Smile2Unlock_v2/build/research/darkface/README.md) 中发现的小脸截断问题：**保留 P2/stride=4 高分辨率浅层分支（推理裁剪或按输入自适应）** 或训练期确保多尺度覆盖 `320 / 480 / 640`，避免像 SeetaFace 固定 min_face=20 那样丢失小脸。

#### 极轻备选方案：`LCF + YuNet-S`（`< 1 MB` FP32 ONNX）
- 若未来硬约束极致 CPU 延迟或 `<1 MB` 体积，直接采用 `ShiqiYu/libfacedetection.train` 的 `YuNet` 加宽版（通道约 `1.5×~2×` 的 `YuNet-S`，带原生 5 点头）前置同样的 `LCF` 模块，并用完全相同的蒸馏与数据配方训练。

### 2. 五路混合训练数据配比（弥补 DarkFace 无 5 点且缺近景的短板）

| 数据源 | 采样权重 | 退化策略 | 损失监督规则（严格隔离） |
|---|---|---|---|
| **1. WIDER FACE Train**（配 RetinaFace 5点标注，~1.28万张） | **40%** | 50% 正常光 + 50% Phase 0 物理弱光/逆光退化；**额外加大 Random Zoom-in 裁剪比例**，刻意造出大量“人脸高占画面 15%~65%”的近景大脸 | 全监督：`Cls + EIoU/DFL + WingLoss(5-pts)` |
| **2. DarkFace Labeled Train Split**（从 6,000 张固定切出 `5,000` 训 / `1,000` 验） | **20%** | 仅做温和几何/缩放/水平翻转（保留真实夜间传感噪声，**不叠加二次极暗压黑**） | **`has_landmarks = 0` 掩码**：只反传 `Cls + BBox`，**完全跳过 5 点损失（不伪造假关键点）** |
| **3. DarkFace Unlabeled (9,000) + Paired (789)** | **15%** | 789 亮暗配对直接做跨光照特征对齐；9,000 无标注暗图走 **Stage 2 EMA Mean-Teacher** | 789 对：配对一致性监督；9,000 图：高置信伪标签框 + 5点（仅置信度 `>0.8` 且几何自洽） |
| **4. 近景高精 5 点人脸集**（300W / WFLW 提 5 点 / CelebA 精选近景，~1.5~2万张） | **15%** | 80% 施加物理弱光/逆光/屏幕冷光/噪声退化 | 重点强监督 **5 点 WingLoss**，防止近距离暗脸关键点漂移 |
| **5. 暗室困难负样本**（COCO 无人像室内背景图，~5,000 张） | **10%** | 重度暗化 + 高 ISO 噪声 + MJPEG 压缩 | 全负样本分类损失，压制高噪点背景误框 |

### 3. 两阶段训练配方（64GB ROCm）
- **预备 Teacher**：用公开高精 `YOLO11l-face-5pt`（或 `RetinaFace-ResNet50`）在 **正常光 WIDER FACE + 300W** 上微调至收敛后冻结，作为正常光锚定教师。
- **Stage 1：有监督微调 + HLA 跨光照特征蒸馏（50 Epochs）**
  - 加载正常光预训练权重初始化，Batch Size `128~256`，多尺度 `[320, 480, 640]` 随机切换，AdamW (`Backbone lr=2e-4`, `Head & LCF lr=1e-3`, `Weight Decay=1e-4`, `Warmup 3 epochs + CosineAnnealing`, 开启 `EMA decay=0.9999`)。
  - **跨光照蒸馏（HLA-Face 思想）**：对所有有正常光原图的训练样本，冻结 Teacher 吃**正常光原图**，Student 吃**弱光退化图**，在 FPN P3/P4/P5 特征层做通道对齐 CWD/Cosine 特征蒸馏损失（权重 `0.5`），强迫 `LCF + Backbone` 从暗噪图中提取出与正常光一致的面部结构特征。
- **Stage 2：Mean-Teacher 半监督精调（30 Epochs）**
  - 冻结 BatchNorm 统计或降低学习率到 `5e-5`，引入 DarkFace 9,000 张无标注暗图：
    - EMA Teacher 吃**轻微提亮（仅线性曝光拉伸或弱增强）图**生成高置信伪标签（框 + 5 点一致性筛选）；
    - Student 吃**原暗图 + 强几何/噪点扰动**学习对齐，让模型吃透真实暗光传感特性。
  - **预计总耗时**：64GB 单卡约 **4~7 小时**。

### 4. Phase 1 离线验收门槛（全部达标才锁定检测模型）
1. **正常光不回退**：`WIDER FACE Val`（Easy / Medium / Hard）相比未做弱光改造的同级基线掉点 **`< 0.5%`**。
2. **弱光 5 点几何精度（核心！）**：在合成物理弱光/逆光 `WIDER Val` 及 `300W-Dark` 上测 **5 点归一化均方根误差 `NME`（以两眼中心距归一化）**：要求 **`< 3.5%`**（正常光要求 `< 2.0%`）。
3. **DarkFace 压力基线对比**：复用仓库已有 [score.py](file:///home/ation_ciger/Projects/Smile2Unlock_v2/build/research/darkface/score.py) 在固定的 `1,000` 张本地 DarkFace Val 上与 SeetaFace `app640`（`0.614%` AP50）同口径对比，预期 AP50 达到显著量级跃升（例如 `> 25%~40%+`）。
4. **真机近景暗光验证集**：单人脸正/侧/眼镜召回率显著超越原版检测器，**多人场景不漏检为单人（防止单脸门控失效）**，无脸暗室空帧误检率（FPPI）`< 0.01`。

---

## 六、Phase 2：弱光跨光照人脸识别模型训练

### 1. 模型选型（2025 SOTA 蒸馏体系）
- **冻结 Teacher（仅驻留 64GB 显存，不部署）**：
  - **主选：`CVLface adaface_vit_base_kprpe_webface12m`**（CVPR 2024+，Hugging Face / `mk-minchul/cvlface` 开源，非商用许可完全覆盖）。它内建 **KP-RPE（关键点相对位置编码）**，是当前公开最强且抗关键点微漂移能力最好的模型之一。
  - *（可选辅佐：同框架下 `AdaFace-IR101-WebFace12M` 作特征维度直接校验）*
- **Student（实际导出 ONNX 部署，112×112 RGB → 512-d L2 特征）**：
  - **主选：`EdgeFace-S`（~3.65M 参数，约 14 MB FP32 ONNX）** 或 **`EdgeFace-XS`（~1.77M 参数，约 7 MB FP32 ONNX）** —— 结合 CNN 局部归纳偏置（ConvNeXt/EdgeNeXt 风格）+ SDTA 轻量全局注意力 + **LoRaLin（低秩线性映射）**，在弱光低画质下鲁棒性显著强于纯深度可分离 CNN。
  - **备选（若算力极受限的老旧 CPU）：`MobileFaceNet (w=1.5)`（~6~8 MB FP32 ONNX）**。

### 2. 训练数据与“对齐误差闭环注入”
1. **快速冒烟链路验证**：**CASIA-WebFace**（49万图 / 1万 ID）—— 64GB 卡仅需 ~1.5 小时即可跑通全套蒸馏代码并验证指标趋势。
2. **正式产品级微调**：**WebFace4M**（420万图 / 20万 ID，清洗极干净且包含丰富暗光、逆光、大姿态样本，非商用完全可用）。
3. **在线孪生视图与关键点误差注入（必做闭环！）**：
   对每张输入 ID 样本同时生成两份对齐后的 `112×112` 图：
   - **`x_clean`（给冻结 Teacher 与 Student 正样本分支）**：正常光原图 + 随机水平翻转 + 轻微 `1 px` 对齐微调；
   - **`x_dark`（专供 Student 强抗退化分支）**：先经过 Phase 0 物理弱光/逆光/噪声/MJPEG 退化（人脸区均值 `luma ≥ 25/255`），再依据 **Phase 1 检测器在真实暗光下实测出的 5 点误差分布（例如 $\sigma \approx 1.5\sim 3.0\text{ px}$ 高斯抖动 + 轻微缩放/旋转误差）**，在 Umeyama 仿射对齐阶段故意注入随机关键点扰动后裁剪 `112×112`！
   > **为什么这步决定成败**：否则离线评测用的是完美 GT 5 点，一接上真实暗光检测器的略微抖动 5 点，识别余弦相似度就会直接跳水。

### 3. 64GB 显存核心训练配方：**AdaFace + AdaDistill 课程动态类中心蒸馏**
直接使用 CVLface 同源优化范式：开启 **BF16（或 FP16）+ Partial FC（随机采样 15%~20% 身份类中心，单卡 64GB 轻松吞下 WebFace4M 20 万 ID）**，从 `EdgeFace` 官方正常光预训练权重初始化。

**联合损失函数设计：**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{AdaFace}}(z_{\text{clean}}, y) + \lambda_1 \mathcal{L}_{\text{AdaFace}}(z_{\text{dark}}, y) + \lambda_2 \mathcal{L}_{\text{AdaDistill}}(z_{\text{dark}}, c^{\text{teacher}}_{\text{clean}}, y) + \lambda_3 \big(1 - \cos(z_{\text{dark}}, z^{\text{teacher}}_{\text{clean}})\big) + \lambda_4 \big(1 - \cos(z_{\text{dark}}, \text{sg}(z_{\text{clean}}))\big)$$

- **为什么用 `AdaFace` + `AdaDistill` 而不是普通 `ArcFace + MSE`**：
  1. **`AdaFace` 质量自适应 Margin**：根据图像特征模长 $\|z\|$（经过 batch 滑动均值/方差标准化）动态调整 angular/additive margin——清晰脸给严苛 margin 拉开类间距，高噪暗图自动放宽 margin，防止过拟合弱光噪点造成跨人误识（FAR 升高）。
  2. **`AdaDistill` 课程动态类中心蒸馏**：传统蒸馏只逼近单个样本点 $z^{\text{teacher}}$；`AdaDistill` 直接利用 Teacher 生成的动态类中心 $c^{\text{teacher}}$（可随训练早期到后期课程化提升蒸馏强度）把学生看暗图的整个身份簇几何结构对齐到 Teacher 的超球面上。
  3. **特征模长 $\|z\|$ 兼作推理质量门控**：极糊/过暗/严重运动模糊帧产生的未归一化特征模长 $\|z\|$ 显著低于清晰帧，可用于离线拒绝废帧录入或误匹配。

- **推荐超参（WebFace4M）**：
  - 初始阶段：前 `1 epoch` 仅放开 `EdgeFace` 最后两个 stage + Head 适应暗光蒸馏，之后全网络解冻微调；
  - Batch Size：**`512` 对（`1024` 张 `112×112` / step）**；
  - 损失权重：$\lambda_1 = 1.0,\ \lambda_2 = 1.5,\ \lambda_3 = 3.0,\ \lambda_4 = 0.5$（$\lambda_2, \lambda_3$ 前 3 epoch 线性升温）；
  - 优化器：AdamW，`lr = 2e-4`（Backbone）/ `5e-4`（Head），`Weight Decay = 5e-4`，CosineAnnealing，总计 **`14 ~ 18 epochs`**，开启 EMA；
  - **预计耗时**：64GB 单卡约 **10 ~ 20 小时**。

### 4. Phase 2 离线验收门槛
1. **正常光防遗忘**：`LFW`、`CFP-FP`、`AgeDB-30`、`CPLFW`、`CALFW` 相比官方原版 `EdgeFace` 权重**不得退化**（偏姿态/跨年龄允许微涨或掉点 `<0.15%`）。
2. **公开低质量权威基准**：`IJB-B` / `IJB-C`（重点看 `TAR @ FAR = 1e-4, 1e-5`）与 `TinyFace`（Rank-1/5），相对原版轻量基线必须有明确正收益。
3. **核心产品协议 — “亮光注册 ↔ 暗光/逆光解锁”（Clean Gallery → Dark Probe）**：
   - 在 **CFP-FP / AgeDB 合成暗光集** 与 **自采真机多光照验证集** 上评测：
     - **使用 Phase 1 弱光检测器实际输出的 5 点对齐** vs **GT 5 点对齐**，两者在 `TAR @ FAR = 1e-4` 下差距必须 **`< 1.0%`**；
     - 画出**亮-亮、亮-暗、异人亮-暗**三组余弦相似度分布直方图，确保同一固定 cosine 阈值在亮光和暗光下均位于安全谷底，无需随亮度切换阈值。

---

## 七、Phase 3：弱光时序活体检测模型训练（最核心、最易踩坑模块）

弱光下的反欺骗是最难的环节：暗光会同时破坏打印纸网点与手机摩尔纹等高频信息，而暗室中自发光手机屏反而偏亮偏蓝。本阶段采用 **2025–2026 最前沿的 `DINOv2 with Registers` 双流时空 Teacher + 时序质量感知轻量学生 `ST-FAS-Lite` + 帧冻结刚性手抖硬负样本机制**。

### 1. Teacher 与 Student 架构设计

#### （1）Teacher（64GB 显存训练，两阶段后冻结，不部署）
- **全局结构语义流**：**`DINOv2 with Registers`（`vit_base_patch14_reg4_dinov2`, `timm` 原生纯 PyTorch 支持，非商用）+ LoRA（rank=16）+ 跨帧时序注意力层**。
  - 遵循 **ICCVW 2025 (`arXiv:2510.17201`) / AAAI 2026** 发现：DINOv2 的 `4` 个 Register tokens 能吸收 Transformer 背景高范数伪影，使注意力精准聚焦于面具边缘、屏幕折光与皮肤微结构；相比 CLIP，在未见摄像头和跨光照条件下具备更强的域泛化力。
- **高频微纹理流（补偿 ViT patch 14×14 空间降采样损失）**：
  - 轻量 **`ConvNeXtV2-Tiny`（输入接入 3 通道 RGB + 3 通道固定拉普拉斯/CDC 高通残差 = 6 通道）**，门控融合到 Teacher 时空头。

#### （2）Student：**`ST-FAS-Lite`（双尺度、质量感知、可流式缓存的时空轻量网络，约 `3~5 MB` FP32 ONNX）**
为了兼顾**强防伪性能**与**未来 CPU 实时帧率（只算当前新帧，不重复算历史 7 帧！）**，设计为 **“逐帧共享空间编码器 + 轻量跨帧时空头”**：

1. **双尺度时空输入茎（Frame $t$ 与相邻帧差 $\Delta I_t = I_t - I_{t-1}$）**：
   - 分支 1（**纹理分支，1.5× 框，`128×128`**）：拼接 `[RGB_t, |RGB_t - RGB_{t-1}|, CDC_highpass(RGB_t)]`（轻量 3×3 中心差分卷积核），捕捉皮肤微观血流/噪点时序演化、屏幕像素网格频闪与纸面纹理；
   - 分支 2（**上下文分支，2.7× 框，`128×128`**）：吃 `[RGB_t]`，捕捉手机屏幕发光边界、外框折反光、纸张硬折痕、头肩背景光照自然过渡。
2. **轻量化空间骨干**：
   - 选用 **`ConvNeXtV2-Atto`（或 `MobileNetV4-Conv-Small`）**（纯深度可分离卷积 + GRN，CPU 友好的硬件访存比），双分支共享或非对称通道（`1.5×` 宽 `1.0`，`2.7×` 宽 `0.5`），单帧输出空间 Patch 特征 $F_t \in \mathbb{R}^{C \times h \times w}$ 与全局向量 $g_t \in \mathbb{R}^{D}$。
   - **推理优势**：历史 7 帧的 $g_{t-7}, \dots, g_{t-1}$ 直接缓存在内存队列里，**每次新帧到来只需对当前 1 帧跑一次 2D 空间骨干！**
3. **跨帧时序建模头（Temporal Head，吃 `T=8` 帧特征序列 $[B, 8, D]$）**：
   - `1 层 Depthwise Temporal Conv (kernel=3)`（捕捉局部帧间突变，如屏闪与微僵死） + **`2 层轻量 Temporal Self-Attention (dim=192, heads=4)`**（纯 PyTorch `SDPA` 实现，建立 8 帧全局动作、闪烁与刚性运动相关性，单次推理开销 `< 0.3ms`）。
4. **FIQA-FAS 帧级质量加权池化 + 多任务防作弊输出头**：
   - **质量分支（Clarity / Quality Gate）**：对每帧输出质量 logit $q_t$，用 `softmax(q_t)` 作为 **FIQA-FAS 注意力权重**对 8 帧特征加权融合，自动抑制被瞬间晃糊或瞬间极黑的废帧；同时输出全局 `clarity_score ∈ [0, 1]`。
   - **主分类头（5 分类细粒度，优于纯二分类）**：
     - `0: Live（真人）`
     - `1: Print（纸质/照片打印）`
     - `2: Replay（屏幕翻拍：手机/平板/显示器）`
     - `3: 3D Mask / Cutout（如有对应数据）`
     - `4: Temporal-Frozen / Rigid-Photo（合成帧冻结+刚性2D平移伪造类）`
     推理时将 `1 - P(Live)` 作为欺骗总分（或直接取 `P(Live)`）。
   - **伪 3D 深度图辅助头（仅训练期挂载，ONNX 导出直接剪除，推理零开销！）**：
     - 真人帧（由离线 PRNet/3DDFA-V2 预先生成的 Face 3D 深度图监督）；所有平面 Print / Replay / Rigid-Photo 帧全监督为 `0` 平面（或仅保留轻微平倾角）。强迫轻量网络学习立体人脸光影结构，而不是背诵屏幕色偏或背景亮度。

### 2. 训练数据集与“严格禁用静态图充数”红线

- **核心时序视频数据集（真实连续视频，必须作为时序阶段主力）**：
  - **OULU-NPU**（重点遵循 Protocol 1 跨光照/跨背景评估与全量训练）；
  - **SiW / SiW-Mv2**（高质量多姿态、多屏幕、多攻击类型视频）；
  - **CASIA-FASD + Replay-Attack + MSU-MFSD**（经典多摄像头跨域基座，压榨摄像头传感器无关性）。
- **静态单帧数据集（仅用于 Stage 1 空间分支预热与 Teacher 对齐，绝不进真视频时序假动作）**：
  - **CelebA-Spoof**（~62万图，带官方光照属性 `Normal/Strong/Back/Dark`，极其珍贵）与 **UniAttackData+ 图像子集**。
  - **⚠️ 严格红线**：**绝对禁止把 CelebA-Spoof 的一张静态真人图片复制 8 份（或轻微平移）当成“真人 Live 视频”喂进时序分支**！否则模型会学到“静止不动 = 真人”，一到真实场景就会放行静态照片！
- **自采真机 4 光照视频小数据集（少而精，性价比之王）**：
  - 按人隔离后，拿出 2~3 人在正常光/逆光/极暗/屏幕光下的真人、不同亮度手机翻拍与打印纸视频并入微调集，弥合旧公开数据集（480p 老手机）与现代高清/宽动态 UVC 摄像头的画质鸿沟。

### 3. 三大实战防翻车核心增广策略

#### 策略 A：物理差异化暗光增广（专治“暗室手机屏绕过”）
| 样本类型 | 物理正确的暗光模拟方式 | 严禁犯的错误 |
|---|---|---|
| **真人视频 (Live)** | 线性降暗 + 强逆光窗背景过曝 / 屏幕冷光打脸 + i.i.d. 逐帧高 ISO 噪点 + `1/15s~1/30s` 轻度运动模糊 + MJPEG 压缩 | 不能把真假样本用同一条曲线压黑 |
| **打印纸 (Print)** | 整体压暗 + **降低局部反差（模拟纸张漫反射灰雾感）** + 轻微偏色 + i.i.d. 传感器噪点 | 不能只保留高清亮光打印纸 |
| **屏幕翻拍 (Replay)** | **重点保留“暗室自发光特征”！**<br>1. 压暗画面**外围环境与人脸边框外区域**，但让人脸区保持**中等或偏亮、冷青白平衡偏色**；<br>2. 叠加轻微摩尔纹周期条纹、屏幕刷新滚动亮暗条纹（60Hz/120Hz 随时间微移动！）与边界外溢光晕；<br>3. 同时按 **30% 比例**保留手机亮度调得很暗（20%亮度）的压暗翻拍样本，双向堵死。 | 严禁无脑压暗所有 Replay，否则暗室高亮度手机一怼摄像头就通过 |

> **必做反捷径体检**：每个 Epoch 统计训练 Batch 内 `Live` 与 `Spoof` 的**人脸区平均亮度（Luma）与局部方差直方图**，验证两组分布高度重叠；如若出现“暗的多为 Live，亮的多为 Spoof”，必须立即调整增广采样比例。

#### 策略 B：基于“时序伪造解耦”的硬负样本合成（彻底压垮静态照片与翻拍视频）
直接从 **真实暗光真人视频** 取第 $k$ 帧 $I_k$，在训练端在线合成两类**亮度分布与真人 100% 一致**的硬负样本（`label = 4: Temporal-Frozen / Rigid-Photo`）：
1. **真帧冻结 + 刚性 2D 仿射微手抖**：把同一张真人脸帧做连续 8 帧微小平移/微旋转/微尺度仿射抖动（模仿手持照片/手机在暗光下轻微手抖），但**画面内眨眼、微表情、真实光影透视变化完全冻结**，同时每帧叠加独立的实时摄像头传感器 i.i.d. 噪点（模拟相机本身正在实时拍摄这张假照片！）。
2. **局部区域透视弯曲模拟纸张轻弯**：对单帧做轻度平滑柱面/网格弹性弯曲，模拟人手拿着打印照片轻微弯折晃动。
> 这套构造强制 `ST-FAS-Lite` 必须学会区分 **“真实 3D 生理微动 + 实时真实噪声”** 与 **“2D 刚性抖动 + 实时真实噪声”**，彻底无法靠亮度、清晰度或单帧噪点量作弊！

#### 策略 C：时序裁剪框随机抖动增广（适配真实检测器前端）
在实际端到端使用时，Phase 1 检测器输出的 `bbox` 在连续 8 帧之间会有 `1%~4%` 的随机坐标抖动。因此在训练读取每段 8 帧视频裁剪 `1.5×` / `2.7×` 区域时，主动向每帧独立注入 **`±2%~4%` 框宽高随机抖动与 `±2%` 尺度抖动**，防止部署时把真人帧因检测框微抖误判成攻击。

### 4. 两阶段训练配方（64GB ROCm）
- **Stage 1：空间预热 & DINOv2-Reg 时空 Teacher 训练（约 6~10 小时）**
  1. 先用 `CelebA-Spoof + 公开视频抽帧 + 差异化物理暗光增广` 对 **Student 空间分支**与 **DINOv2-Reg + ConvNeXtV2 Teacher** 进行单帧预热（引入 **SSDG 单侧域泛化对比损失**，强制所有相机/光照下的真人空间紧凑聚拢，各攻击类型推开；叠加 **Patch-wise 遮挡/局部混合增广**）。
  2. 接入 `T=8` 帧真实视频数据集 + 策略 B 硬负样本，微调 Teacher 的 `LoRA` 与 `Temporal Transformer` 收敛至顶尖跨域水平。
- **Stage 2：时空多模态蒸馏到 `ST-FAS-Lite`（约 6~10 小时）**
  - 冻结 Teacher，每个 Batch 严格按 **`1:1`（Live : 各类 Spoof + 硬负样本）** 均衡采样多源视频片段（Batch Size `64~128` 片段 = `512~1024` 帧/step，BF16，AdamW `lr=3e-4`, Cosine, 40~60 Epochs, EMA）。
  - **复合蒸馏与监督损失**：
    $$\mathcal{L}_{\text{FAS}} = \mathcal{L}_{\text{Focal-CE}}(y_{\text{5cls}}) + 0.5\mathcal{L}_{\text{Depth}}(D_{\text{pred}}, D_{\text{pseudo}}) + 0.3\mathcal{L}_{\text{Clarity}} + 1.0\mathcal{L}_{\text{KD-Logits}} + 1.0\mathcal{L}_{\text{Patch-Affinity-KD}}$$
    其中 **`Patch-Affinity-KD`**：将 Student `1.5×` 主干输出的 Space-Time Patch 自相似矩阵对齐到 `DINOv2-with-Registers` 最后一层的纯净 Patch 亲和度矩阵（既规避维度不同，又把大模型关注微纹理和结构的能力精准转移给轻量网络）。

### 5. Phase 3 离线验收门槛（遵循 ISO/IEC 30107-3 分层矩阵）
绝不允许只看一个总 Accuracy！必须打印 **4 档光照 × 4 类攻击** 分层评测表：
- 核心指标：**APCER**（攻击错放率）、**BPCER**（真人误拒率）、**ACER = (APCER + BPCER) / 2**、**AUC**、**HTER**。
- **产品级红线评测法**：固定全场景统一阈值于 **`APCER ≤ 1.0%`**（安全线）与 **`APCER ≤ 2.0%`**（参考线），逐一检查以下切片的 `BPCER` 与特定攻击通过率：
  1. **正常光**（期望 BPCER `< 1%`）
  2. **强逆光**（期望 BPCER `< 3%~5%`）
  3. **暗室微光 / 仅屏幕补光**（期望 BPCER `< 5%~8%`）
  4. **高危专项压测**：暗室高亮（100%）手机翻拍、暗室低亮（20%）手机翻拍、暗室打印纸弯折、真帧冻结+手抖片段。
- **跨域 Leave-One-Out 检验**：遵循 `OULU + CASIA + Replay → 测试 MSU`（以及 `OULU-NPU Protocol 1` 跨光照）验证没有过拟合到单一相机设备。

---

## 八、Phase 4：离线端到端 ONNX 串联验收、量化与归档

本阶段依然**纯 Python + `onnxruntime` 在离线视频验证集上完成**，不改动主仓库源码：

### 1. 流式与整段 ONNX 等价性核对 + 数值对齐
- 将 `Model 1`、`Model 2`、`Model 3（整段版 + 拆分的空间单帧版/时序头版）` 导出为标准 **ONNX (opset 17/18)**；
- 验证 **PyTorch vs ONNX FP32** 全链路输出最大绝对误差 `< 1e-4`；
- 验证 **8 帧整段版活体 ONNX** vs **逐帧缓存空间特征的流式双 ONNX** 输出概率完全一致（误差 `< 1e-4`）。

### 2. 三模型真机视频流串联闭环体检
按真实运行逻辑跑通全流程：
`未增强原视频帧 → ONNX 检测(含内部LCF) + 5点 → 单脸检查 → 原图 Umeyama 112×112 对齐 → ONNX 识别特征 + ||z||；同时裁剪 1.5× & 2.7× 128×128 → ONNX 流式活体缓存 → 8帧时序头出 live_prob 与 clarity`
1. **端到端误差传递核查**：检查检测器输出 5 点在暗光下的抖动，是否引起识别 `TAR@FAR=1e-4` 掉点超过 `1%`，以及检测框抖动是否导致活体 `BPCER` 明显上升。
2. **统一阈值冻结**：找到一组在**正常光 / 强逆光 / 暗室屏幕光**三档条件下全部生效的推荐基线参考阈值 `(det_score_th, rec_cosine_th, rec_norm_min, liveness_th, clarity_th)`。
3. **可选轻量化 / 量化尝试（仅当体积或 CPU 延迟仍不满足时）**：
   - 若某模块需要 INT8 压缩，**切忌直接对 `LCF` 或活体高频层暴力 PTQ（容易把暗噪量化崩）**；应优先保持 `FP16` 存储或做针对性 **QAT（量化感知训练，5~10 epochs）**，并在真机弱光验证集重跑一遍全链路对比。
4. **普通 x86 CPU 耗时摸底**：在一台普通 x86_64 CPU（分别测试 1 线程与 4 线程 `onnxruntime` CPU EP）统计每个模型的 P50/P95 单帧延迟、模型文件总体积、加载时间与峰值常驻内存（RSS）。

### 3. 可复现产物归档清单（交付物）
训练工程目录须产出完整可追溯归档包（**不直接把巨型数据集和未审核权重提交到应用主仓**）：
- `models/`：`detector_5pt_lcf.onnx`、`recognizer_edgeface.onnx`、`fas_spatial_enc.onnx`、`fas_temporal_head.onnx` 及各自 `SHA-256`。
- `manifest.json`：每个模型的输入输出张量名/shape、通道顺序（`RGB`）、归一化公式、5 点顺序、112×112 对齐模板坐标、双尺度裁剪比例（`1.5× / 2.7×`）、时序长度（`T=8`）与推荐离线阈值。
- `eval_report.md`：包含 DarkFace 1,000 Val 同口径 `score.py` 对照、WIDER / IJB / Clean-to-Dark 识别曲线、ISO 30107-3 光照×攻击分层矩阵、流式 vs 整段校验误差、x86 CPU P50/P95 延迟表。
- `provenance.md`：训练代码 commit、PyTorch & ROCm 版本、各数据集 split hash、预训练源权重 SHA-256 与非商用许可记录。

---

## 九、独立训练工程目录建议与总排期

建议在应用仓库之外（或 Git 忽略工作目录中）新建独立的训练工坊目录，保持主仓库零污染：

```text
face_lowlight_lab/
├── configs/                 # detector / recognizer / liveness_st YAML 配置
├── env_check_rocm.py        # Phase 0: 5 分钟 ROCm BF16/FP16/SDPA 压测脚本
├── common/
│   ├── degradations.py      # GPU 矢量物理级弱光/逆光/噪声/YUV-MJPEG 退化引擎
│   ├── align.py             # 唯一权威 5-pt Umeyama 112x112 与 1.5x/2.7x 裁剪实现
│   └── metrics.py           # NME / TAR@FAR / ISO 30107-3 APCER-BPCER-ACER 统计
├── detector/                # Phase 1: LCF + YOLO11n-face-5pt + Mean-Teacher
├── recognizer/              # Phase 2: CVLface KP-RPE Teacher + AdaDistill -> EdgeFace
├── liveness/                # Phase 3: DINOv2-Reg Teacher + ST-FAS-Lite 时序蒸馏
├── eval_e2e_onnx.py         # Phase 4: 纯 onnxruntime 视频流闭环验收 + CPU 压测
└── artifacts/               # 导出的 ONNX 模型、SHA-256、manifest 与评测报告
```

### 执行排期表（基于单台 64GB ROCm 服务器）

| 阶段 | 核心任务与产出 | 预估周期 | 阶段卡点（不通过不往下走） |
|---|---|---|---|
| **Phase 0** | ROCm 算子/精度压测 + GPU 物理弱光引擎 + 自采 4 光照真机验证集标注 | **1 ~ 1.5 天** | ROCm BF16/FP16 跑通无报错；退化样本肉眼贴近真机暗光图 |
| **Phase 1** | 训练 `LCF + YOLO11n-face-5pt`（Stage 1 蒸馏 + Stage 2 DarkFace 9,000 半监督） | **1.5 ~ 2 天** | 正常光掉点 `<0.5%`；暗光 5点 `NME < 3.5%`；DarkFace 远超 0.614% 基线 |
| **Phase 2** | CASIA 冒烟验证 → WebFace4M `CVLface KP-RPE` + `AdaDistill` 蒸馏 `EdgeFace` | **2 ~ 3 天** | 正常光基准不跌；暗光自动对齐 vs GT 对齐 `TAR@FAR=1e-4` 掉点 `<1%` |
| **Phase 3** | `DINOv2-Reg` 双流 Teacher → `ST-FAS-Lite`（含冻结手抖负样本 & 差异化增广） | **2.5 ~ 3.5 天** | `APCER ≤ 1~2%` 下暗光/逆光真人 BPCER 达标；暗室亮屏/暗屏均拦截 |
| **Phase 4** | 全链路 ONNX（含流式缓存等价验证）闭环验收 + CPU P50/P95 压测 + 归档 | **1 天** | 统一阈值跨 3 档光照稳定通过；生成完整报告与 SHA-256 清单 |
