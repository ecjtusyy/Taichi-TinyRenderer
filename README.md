Taichi TinyRenderer（软栅格渲染器）

一个用 Python + Taichi 写的极简软渲染器，参考 TinyRenderer 思路，包含：

相机：LookAt / 透视投影 / 视口变换

几何处理：背面剔除

光栅化：三角形重心坐标填充 + Z-Buffer

着色：Lambert 漫反射（可调方向光）

纹理（可选）：最近邻采样（当前为屏幕空间插值，后续可升级为严格透视校正）

目标：学习图形学基础、搭建一个干净可扩展的教学/实验骨架，并在多后端（Vulkan/CUDA/OpenGL/CPU）上运行。

目录

环境与安装

快速上手（无纹理 Demo）

带纹理的使用方式

命令行参数说明

常见问题与排查

项目结构建议

Roadmap

许可证

环境与安装

Python ≥ 3.8

推荐 Taichi ≥ 1.6（1.7 也可）

其他依赖：numpy

# 安装 taichi（会带上 numpy）
pip install taichi


本项目内置了“多后端初始化”逻辑：按 Vulkan → CUDA → OpenGL → CPU 的优先级尝试；能跑哪个就用哪个，因此无需手动判断 cuda_available()。

快速上手（无纹理 Demo）

准备一个最小三角形 OBJ（复制下面到终端执行）：

cat > single_triangle.obj << 'EOF'
# Single triangle in XY plane (Z=0)
v  -0.8 -0.6  0.0
v   0.8 -0.6  0.0
v   0.0  0.7  0.0
f 1 2 3
EOF


运行渲染（输出 triangle_out.png）：

python main.py --obj single_triangle.obj --out triangle_out.png \
  --width 512 --height 512 \
  --eye 0 0 3 --center 0 0 0 --up 0 1 0


看到终端打印 [OK] Saved image to triangle_out.png 就成功了。

带纹理的使用方式

需要一个带 vt 的 OBJ 与一张纹理图片（PNG/JPG）。

python main.py \
  --obj path/to/your_model_with_vt.obj \
  --texture path/to/diffuse.png \
  --out textured_out.png \
  --width 800 --height 800 \
  --eye 1 1 3 --center 0 0 0 --up 0 1 0


说明：当前纹理坐标在片元阶段做线性插值（足够应付大多数测试），如需**严格透视校正（uv/w）**与 Phong 法线插值，见下方 Roadmap。

命令行参数说明
--obj       必填，OBJ 模型路径（支持 v / vt / f；多边形面自动扇形三角化）
--texture   可选，纹理图片路径（PNG/JPG），需要 OBJ 含 vt
--out       输出图片名，默认 out.png
--width     图像宽，默认 800
--height    图像高，默认 800
--fov       垂直视角（度），默认 60
--near      近裁剪面，默认 0.1
--far       远裁剪面，默认 100
--eye       相机位置，三个数
--center    相机看向点，三个数
--up        相机上方向，三个数
--bg        背景色 RGB，默认 0.05 0.05 0.08
--light     方向光（世界空间）向量，默认 0.5 1.0 0.3


实际渲染流程：

LookAt 生成视图矩阵 → Perspective 生成投影矩阵 → 得到 PV

顶点经 PV 变换到 NDC，再用 Viewport 映射到屏幕像素坐标

背面剔除 → 包围盒扫描 → 重心坐标判断覆盖 → 深度测试 → 着色/采样

常见问题与排查

Q1: 初始化就报错 / 后端不可用

本项目会自动按 Vulkan → CUDA → OpenGL → CPU 依次尝试；

如果 GPU/驱动环境不完整，最终会回退到 CPU；

若全部失败，检查：显卡驱动、vulkaninfo（如需要）、是否有显示环境（OpenGL 通常需要）。

Q2: 输出全黑/很暗

尝试把光方向改得更“朝向相机”：--light 0 0 1；

模型法线是用三角面三个世界坐标叉乘得到的面法线，如果面朝背面会被剔除或亮度很低。

Q3: 有纹理但看不到

确认 OBJ 包含 vt，且命令行传入了 --texture path/to/diffuse.png；

纹理坐标 V 轴在采样函数里做了翻转，如果出现颠倒，先看纹理本身坐标约定。

Q4: OBJ 读不出来

目前的加载器是“最小可用”版本，仅支持常见的 v/vt/f；复杂特性（法线、材质、mtl 引用等）未使用。

多边形面会自动扇形三角化。

项目结构建议

最小结构即可跑：

.
├── main.py                  # 渲染器主程序（Taichi 内核 + OBJ 加载 + 管线拼装）
├── single_triangle.obj      # 可选：最小 Demo（无纹理）
└── README.md                # 本文件


如果你加了纹理和更多模型：

.
├── main.py
├── assets/
│   ├── models/xxx.obj
│   └── textures/diffuse.png
└── README.md

Roadmap

 严格透视校正纹理（插值 uv/w 与 1/w，片元阶段还原）

 Phong 法线插值（解析 vn，顶点法线重心插值 + 归一化）

 线框模式 / 深度图 / UV 可视化（调试与教学更直观）

 阴影映射（Shadow Map）

 Tile-based 并行栅格化（更好的缓存局部性与并行度）

 OBJ/MTL 更完整解析（漫反射、镜面、法线贴图等）

如果你需要，我可以在当前代码上提供最小增量补丁来实现上述任一条。

许可证

建议使用 MIT License（开源友好，可商用）。如你希望我生成一份标准 LICENSE 文件，也可以说一声我直接给你贴上。

引用与致谢

学习路线参考 GAMES101/102/201

实现思路参考 TinyRenderer（教学用）

框架与并行后端由 Taichi 提供
