# 🎨 Taichi TinyRenderer（软栅格渲染器）

一个用 **Python + Taichi** 编写的极简软渲染器，参考 **TinyRenderer** 思路，包含以下核心功能：

- **相机**：LookAt / 透视投影 / 视口变换  
- **几何处理**：背面剔除  
- **光栅化**：三角形重心坐标填充 + Z-Buffer 深度测试  
- **着色**：Lambert 漫反射（可调方向光）  
- **纹理（可选）**：最近邻采样（当前为屏幕空间插值，后续计划支持透视校正）  

> 🎯 目标：学习图形学基础、搭建一个干净可扩展的教学/实验骨架，并在多后端（Vulkan / CUDA / OpenGL / CPU）上运行。

---

## 📂 目录

- [环境与安装](#环境与安装)
- [快速上手（无纹理 Demo）](#快速上手无纹理-demo)
- [带纹理的使用方式](#带纹理的使用方式)
- [命令行参数说明](#命令行参数说明)
- [常见问题与排查](#常见问题与排查)
- [项目结构建议](#项目结构建议)
- [Roadmap](#roadmap)
- [许可证](#许可证)
- [引用与致谢](#引用与致谢)

---

## 🧩 环境与安装

**依赖环境：**
- Python ≥ 3.8  
- 推荐 **Taichi ≥ 1.6**（1.7 也可）  
- 其他依赖：`numpy`

安装 Taichi（会自动安装 numpy）：
```bash
pip install taichi
````

> 本项目内置了“多后端初始化”逻辑：按 **Vulkan → CUDA → OpenGL → CPU** 的优先级尝试；
> 能跑哪个就自动用哪个，因此无需手动判断 `cuda_available()`。

---

## 🚀 快速上手（无纹理 Demo）

准备一个最小三角形 OBJ：

```bash
cat > single_triangle.obj << 'EOF'
# Single triangle in XY plane (Z=0)
v  -0.8 -0.6  0.0
v   0.8 -0.6  0.0
v   0.0  0.7  0.0
f 1 2 3
EOF
```

运行渲染：

```bash
python main.py --obj single_triangle.obj --out triangle_out.png \
  --width 512 --height 512 \
  --eye 0 0 3 --center 0 0 0 --up 0 1 0
```

看到终端打印：

```
[OK] Saved image to triangle_out.png
```

即表示渲染成功。

---

## 🖼 带纹理的使用方式

需要一个带 `vt` 的 OBJ 与一张纹理图片（PNG / JPG）：

```bash
python main.py \
  --obj path/to/your_model_with_vt.obj \
  --texture path/to/diffuse.png \
  --out textured_out.png \
  --width 800 --height 800 \
  --eye 1 1 3 --center 0 0 0 --up 0 1 0
```

> 💡 说明：当前纹理坐标在片元阶段做**线性插值**（足够应付大多数测试），
> 如需 **严格透视校正（uv/w）** 与 **Phong 法线插值**，见下方 Roadmap。

---

## ⚙️ 命令行参数说明

| 参数                     | 说明                                        |
| ---------------------- | ----------------------------------------- |
| `--obj`                | **必填**，OBJ 模型路径（支持 v / vt / f；多边形自动扇形三角化） |
| `--texture`            | 可选，纹理图片路径（PNG/JPG，需要 OBJ 含 vt）            |
| `--out`                | 输出图片名，默认 `out.png`                        |
| `--width` / `--height` | 输出图像尺寸，默认 800×800                         |
| `--fov`                | 垂直视角（度），默认 60                             |
| `--near` / `--far`     | 近裁剪面 / 远裁剪面，默认 0.1 / 100                  |
| `--eye`                | 相机位置（x y z）                               |
| `--center`             | 相机看向点（x y z）                              |
| `--up`                 | 相机上方向（x y z）                              |
| `--bg`                 | 背景色 RGB，默认 `0.05 0.05 0.08`               |
| `--light`              | 方向光向量（世界空间），默认 `0.5 1.0 0.3`              |

---

## 🧠 实际渲染流程

1. **LookAt** → 生成视图矩阵
2. **Perspective** → 生成投影矩阵
3. 得到 PV（Projection × View）矩阵
4. 顶点变换到 NDC，再映射到屏幕像素坐标
5. 背面剔除
6. 包围盒扫描 + 重心坐标判断覆盖
7. 深度测试 + 漫反射着色（可采样纹理）

---

## ❓ 常见问题与排查

**Q1：初始化时报错 / 后端不可用**
👉 自动尝试后端顺序为 Vulkan → CUDA → OpenGL → CPU；
如全部失败，请检查：

* 显卡驱动；
* `vulkaninfo` 输出；
* 是否有显示环境（OpenGL 需图形环境）。

---

**Q2：输出全黑 / 很暗**

* 试试调整光照方向：`--light 0 0 1`；
* 模型法线是通过三角面世界坐标叉乘获得，背面会被剔除或亮度低。

---

**Q3：加载了纹理但看不到**

* 确认 OBJ 含 `vt`；
* 命令行传入了 `--texture`；
* 纹理 V 轴在采样函数中做了翻转，如颠倒请检查纹理坐标方向。

---

**Q4：OBJ 读不出来**

* 当前加载器为**最小可用版**，支持 v / vt / f；
* 不支持复杂特性（法线、材质、MTL 文件）；
* 多边形面会自动**扇形三角化**。

---

## 📁 项目结构建议

### 最小结构（即可运行）

```
.
├── main.py                  # 渲染器主程序（Taichi 内核 + OBJ 加载 + 管线拼装）
├── single_triangle.obj      # 可选：最小 Demo（无纹理）
└── README.md                # 本文件
```

### 含纹理与模型的推荐结构

```
.
├── main.py
├── assets/
│   ├── models/xxx.obj
│   └── textures/diffuse.png
└── README.md
```

---

## 🧭 Roadmap

* [ ] 严格透视校正纹理（插值 uv/w 与 1/w，片元阶段还原）
* [ ] Phong 法线插值（解析 vn，顶点法线重心插值 + 归一化）
* [ ] 线框模式 / 深度图 / UV 可视化（调试与教学）
* [ ] 阴影映射（Shadow Map）
* [ ] Tile-based 并行栅格化（提高缓存局部性与并行度）
* [ ] 更完整的 OBJ/MTL 支持（漫反射、镜面、法线贴图等）

> 💡 如果你需要，我可以提供最小增量补丁来实现上述任意一条。

---

## 📜 许可证

建议使用 **MIT License**（开源友好，可商用）。
如需生成标准 LICENSE 文件，可直接使用以下模板：

```text
MIT License
Copyright (c) 2025 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 引用与致谢

* 学习路线参考：[GAMES101 / 102 / 201](https://games-cn.org/)
* 实现思路参考：**TinyRenderer（教学用）**
* 框架与并行后端支持：**Taichi**

---

```
```
