import argparse
import math
import os
from typing import List, Tuple, Optional

import numpy as np
import taichi as ti

# ------------------------------------------------------------
# 这里做一个 Taichi 的初始化函数
# 思路：按顺序尝试几个后端（Vulkan -> CUDA -> OpenGL -> CPU）
# 哪个能用就用哪个，这样就不用判断 cuda_available 之类的了
# ------------------------------------------------------------
def init_taichi():
    """
    尝试多个后端，顺序：Vulkan -> CUDA -> OpenGL -> CPU。
    这样比较稳，不会因为某个后端不可用就直接报错。
    """
    tried = []
    for arch in [getattr(ti, "vulkan", None),
                 getattr(ti, "cuda",   None),
                 getattr(ti, "opengl", None),
                 getattr(ti, "cpu",    None)]:
        if arch is None:
            continue
        try:
            ti.init(arch=arch)
            print(f"[Taichi] Using backend: {arch}")
            return
        except Exception as e:
            tried.append((arch, str(e)))
    msg = " ; ".join([f"{a}: {err}" for a, err in tried]) if tried else "no backends?"
    raise RuntimeError(f"Taichi initialization failed -> {msg}")

init_taichi()


# ---------------------------
# 一个很简单的 OBJ 加载器（在 CPU 端）
# 只支持 v / vt / f，面如果是多边形我就用扇形三角化
# ---------------------------
def load_obj(obj_path: str):
    """
    很简陋的 OBJ 读取：
    - 支持 v 顶点、vt 纹理坐标、f 面
    - f 里面的索引可能是 v、v/vt、v//vn、v/vt/vn，我只取 v 和 vt
    - 多边形面用扇形三角化成三角形
    返回：
      V:  顶点 (N,3)
      VT: 纹理坐标 (T,2)（可能为空）
      F:  三角形索引 (F,3)
      F_vt: 三角形的 vt 索引 (F,3)，如果没有就是 -1
    """
    vs: List[Tuple[float, float, float]] = []
    vts: List[Tuple[float, float]] = []
    faces_v: List[Tuple[int, int, int]] = []
    faces_vt: List[Tuple[int, int, int]] = []

    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                # 顶点
                x, y, z = map(float, parts[1:4])
                vs.append((x, y, z))
            elif tag == "vt" and len(parts) >= 3:
                # 纹理坐标
                u, v = map(float, parts[1:3])
                vts.append((u, v))
            elif tag == "f":
                # 面（可能是3个点，也可能更多，我都三角化）
                fv = []
                fvt = []
                for p in parts[1:]:
                    toks = p.split("/")
                    vi = int(toks[0]) - 1
                    fv.append(vi)
                    if len(toks) >= 2 and toks[1] != "":
                        vti = int(toks[1]) - 1
                        fvt.append(vti)
                    else:
                        fvt.append(-1)
                # 扇形三角化：v0, v[i], v[i+1]
                for i in range(1, len(fv) - 1):
                    faces_v.append((fv[0], fv[i], fv[i + 1]))
                    faces_vt.append((fvt[0], fvt[i], fvt[i + 1]))

    V = np.array(vs, dtype=np.float32) if vs else np.zeros((0, 3), dtype=np.float32)
    VT = np.array(vts, dtype=np.float32) if vts else np.zeros((0, 2), dtype=np.float32)
    F = np.array(faces_v, dtype=np.int32) if faces_v else np.zeros((0, 3), dtype=np.int32)
    F_vt = np.array(faces_vt, dtype=np.int32) if faces_vt else np.full((0, 3), -1, dtype=np.int32)
    return V, VT, F, F_vt


# ---------------------------
# 一些数学小工具（还是在 CPU 端）
# ---------------------------
def look_at(eye, center, up):
    """
    生成一个右手系的 LookAt 视图矩阵。
    就是把相机的位置、看向哪里、上方向，变成一个 4x4 矩阵。
    """
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    z = eye - center
    z /= (np.linalg.norm(z) + 1e-8)
    x = np.cross(up, z)
    x /= (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)

    M = np.eye(4, dtype=np.float32)
    M[0, :3] = x
    M[1, :3] = y
    M[2, :3] = z
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    return M @ T


def perspective(fovy_deg, aspect, z_near, z_far):
    """
    透视投影矩阵。
    fovy 是垂直视角（度），aspect 是宽高比，near/far 是裁剪面。
    """
    fovy = math.radians(fovy_deg)
    f = 1.0 / math.tan(fovy / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (z_far + z_near) / (z_near - z_far)
    M[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    M[3, 2] = -1.0
    return M


def viewport(w, h):
    """
    视口变换，把 NDC 的 [-1,1] 范围映射到像素坐标 [0,w) x [0,h)。
    注意我把 y 反过来，这样图像的上边对应 y=0，比较符合常见图像坐标。
    """
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = w / 2.0
    M[0, 3] = (w - 1) / 2.0
    M[1, 1] = -h / 2.0  # 这里取负号是为了把 y 翻转
    M[1, 3] = (h - 1) / 2.0
    # z 我就保持 NDC 的 z，不做 0..1 映射了，简单点
    return M


# ---------------------------
# 用 Taichi 写的一个很基础的“软渲染器”
# 思路：三角形重心栅格化 + 深度测试 + 简单的漫反射光照
# ---------------------------
@ti.data_oriented
class TinyRendererTi:
    def __init__(self, width: int, height: int, num_faces: int):
        self.W = width
        self.H = height
        self.num_faces = num_faces

        # 帧缓冲和深度缓冲
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=(self.W, self.H))
        self.depth = ti.field(dtype=ti.f32, shape=(self.W, self.H))

        # 每个三角形我们存一下屏幕空间的三个顶点（x,y,z）
        # 以及 ndc 的 z（有时候插值用），还有世界空间顶点（算法线）
        self.tri = ti.Vector.field(3, dtype=ti.f32, shape=(num_faces, 3))
        self.tri_ndc_z = ti.Vector.field(3, dtype=ti.f32, shape=(num_faces, 3))
        self.tri_world = ti.Vector.field(3, dtype=ti.f32, shape=(num_faces, 3))

        # 不管有没有纹理坐标，这里都先分配一个 tri_uv，避免编译期没有这个字段
        self.tri_uv = ti.Vector.field(2, dtype=ti.f32, shape=(num_faces, 3))

        # 光照方向（世界空间）
        self.light_dir = ti.Vector.field(3, dtype=ti.f32, shape=())

    @ti.kernel
    def clear(self, r: ti.f32, g: ti.f32, b: ti.f32):
        """清屏：把颜色设成背景色，深度设成一个很大的值。"""
        for x, y in self.color:
            self.color[x, y] = ti.Vector([r, g, b])
            self.depth[x, y] = 1e9

    @staticmethod
    @ti.func
    def edge_fn(a, b, c):
        """用 2D 坐标算有向面积（其实就是重心算法里常用的边函数）。"""
        return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)

    @ti.func
    def sample_texture(self, tex: ti.types.ndarray(dtype=ti.f32, ndim=3),
                       uv: ti.types.vector(2, ti.f32)) -> ti.types.vector(3, ti.f32):
        """最简单的最近邻采样，uv 范围按照 0..1 来，v 我做了翻转。"""
        H = tex.shape[0]
        W = tex.shape[1]
        u = ti.min(ti.max(uv.x, 0.0), 1.0)
        v = ti.min(ti.max(uv.y, 0.0), 1.0)
        ix = ti.cast(u * (W - 1), ti.i32)
        iy = ti.cast((1.0 - v) * (H - 1), ti.i32)  # 翻转 V
        return ti.Vector([tex[iy, ix, 0], tex[iy, ix, 1], tex[iy, ix, 2]])

    @ti.kernel
    def draw_triangles(self,
                       tex: ti.types.ndarray(dtype=ti.f32, ndim=3),
                       use_texture: ti.i32):
        """
        光栅化所有三角形：
        - 先算包围盒
        - 再用重心坐标判断像素在不在三角形里
        - 通过 Z-Buffer 做可见性
        - 漫反射（Lambert）做个最简单的光照
        """
        for f in range(self.num_faces):
            a = self.tri[f, 0]
            b = self.tri[f, 1]
            c = self.tri[f, 2]

            # 先声明 bbox 的上下界（Taichi 要求变量在所有路径上都有定义）
            min_x = ti.cast(0, ti.i32)
            min_y = ti.cast(0, ti.i32)
            max_x = ti.cast(-1, ti.i32)
            max_y = ti.cast(-1, ti.i32)

            area = TinyRendererTi.edge_fn(a, b, c)
            if area != 0:
                # 算三角形包围盒，并裁剪到屏幕范围
                ix0 = ti.cast(ti.floor(a.x), ti.i32); iy0 = ti.cast(ti.floor(a.y), ti.i32)
                ix1 = ti.cast(ti.floor(b.x), ti.i32); iy1 = ti.cast(ti.floor(b.y), ti.i32)
                ix2 = ti.cast(ti.floor(c.x), ti.i32); iy2 = ti.cast(ti.floor(c.y), ti.i32)
                ax0 = ti.cast(ti.ceil(a.x),  ti.i32); ay0 = ti.cast(ti.ceil(a.y),  ti.i32)
                ax1 = ti.cast(ti.ceil(b.x),  ti.i32); ay1 = ti.cast(ti.ceil(b.y),  ti.i32)
                ax2 = ti.cast(ti.ceil(c.x),  ti.i32); ay2 = ti.cast(ti.ceil(c.y),  ti.i32)

                min_x = ti.max(0, ti.min(ti.min(ix0, ix1), ix2))
                min_y = ti.max(0, ti.min(ti.min(iy0, iy1), iy2))
                max_x = ti.min(self.W - 1, ti.max(ti.max(ax0, ax1), ax2))
                max_y = ti.min(self.H - 1, ti.max(ti.max(ay0, ay1), ay2))

                # 用世界空间的三个顶点算一个面法线，拿来做简单的 Lambert 光照
                wa = self.tri_world[f, 0]
                wb = self.tri_world[f, 1]
                wc = self.tri_world[f, 2]
                n = (wb - wa).cross(wc - wa).normalized()
                L = self.light_dir[None].normalized()
                lambert = ti.max(0.0, n.dot(L))

                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        p = ti.Vector([ti.cast(x, ti.f32) + 0.5, ti.cast(y, ti.f32) + 0.5, 0.0])
                        w0 = TinyRendererTi.edge_fn(b, c, p)
                        w1 = TinyRendererTi.edge_fn(c, a, p)
                        w2 = TinyRendererTi.edge_fn(a, b, p)

                        inside = (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0)
                        if inside:
                            w0f = w0 / area
                            w1f = w1 / area
                            w2f = w2 / area

                            # z 要先声明一个初值（不然 Taichi 会说没定义）
                            z = ti.cast(0.0, ti.f32)
                            if use_texture == 1:
                                # 有纹理时我用 ndc.z 来插值深度（简单点）
                                za = self.tri_ndc_z[f, 0].z
                                zb = self.tri_ndc_z[f, 1].z
                                zc = self.tri_ndc_z[f, 2].z
                                z = w0f * za + w1f * zb + w2f * zc
                            else:
                                # 没纹理就直接用屏幕空间的 z
                                z = w0f * a.z + w1f * b.z + w2f * c.z

                            if z < self.depth[x, y]:
                                base_color = ti.Vector([0.7, 0.7, 0.7])
                                if use_texture == 1:
                                    # 这里的纹理坐标我直接线性插值（严格的透视校正以后可以再加）
                                    uv0 = self.tri_uv[f, 0]
                                    uv1 = self.tri_uv[f, 1]
                                    uv2 = self.tri_uv[f, 2]
                                    uv = uv0 * w0f + uv1 * w1f + uv2 * w2f
                                    base_color = self.sample_texture(tex, uv)
                                color = base_color * (0.2 + 0.8 * lambert)  # 给点环境光，不然太黑
                                self.color[x, y] = color
                                self.depth[x, y] = z

    def render(self,
               verts_world: np.ndarray,
               faces: np.ndarray,
               PV: np.ndarray,
               VP: np.ndarray,
               texcoords: Optional[np.ndarray],
               faces_vt: Optional[np.ndarray],
               light_dir=(0.5, 1.0, 0.3),
               tex_img: Optional[np.ndarray] = None):
        """
        把 CPU 上的数据准备好，然后喂给 Taichi 的字段，再调用 kernel 去画。
        参数大概意思：
          - verts_world: 模型的世界空间坐标
          - faces: 三角形索引
          - PV: 投影 * 视图（先把点变到裁剪空间，再做 NDC）
          - VP: 视口矩阵（把 NDC 映射到屏幕像素）
          - texcoords / faces_vt: 如果模型带 UV，就能采样纹理
          - light_dir: 光方向（世界空间）
          - tex_img: 纹理图（H,W,3），范围 0..1
        """
        F = faces.shape[0]

        # 这些是要传给 Taichi 的中间结果
        tri = np.zeros((F, 3, 3), dtype=np.float32)        # 屏幕坐标 (x,y,z)
        tri_ndc_z = np.zeros((F, 3, 3), dtype=np.float32)  # NDC (x,y,z)，主要用它的 z
        tri_world = np.zeros((F, 3, 3), dtype=np.float32)  # 世界坐标（算法线）
        tri_uv = np.zeros((F, 3, 2), dtype=np.float32)     # UV（即使不用也先给个零）

        # 看看这次渲染要不要用纹理
        use_tex = 0
        if (texcoords is not None) and (faces_vt is not None) and (tex_img is not None):
            use_tex = 1

        # 逐面处理：PV 后做 NDC，再用 VP 变到屏幕
        for i in range(F):
            idx = faces[i]
            P = verts_world[idx]  # 三个顶点
            tri_world[i] = P

            clip_pv = []
            for k in range(3):
                v = np.array([P[k, 0], P[k, 1], P[k, 2], 1.0], dtype=np.float32)
                vc = PV @ v
                clip_pv.append(vc)
            clip_pv = np.stack(clip_pv, axis=0)

            ndc = clip_pv[:, :3] / np.maximum(1e-8, clip_pv[:, 3:4])  # 防止除零
            tri_ndc_z[i] = ndc

            # 视口变换（把 NDC 映射到屏幕像素坐标）
            scr = []
            for k in range(3):
                vndc = np.array([ndc[k, 0], ndc[k, 1], ndc[k, 2], 1.0], dtype=np.float32)
                vscr = VP @ vndc
                scr.append(vscr[:3])
            tri[i] = np.stack(scr, axis=0)

            # 如果要用纹理，就把 UV 填好
            if use_tex == 1:
                vt_idx = faces_vt[i]
                uv_ok = (vt_idx >= 0).all()
                if uv_ok:
                    tri_uv[i, 0] = texcoords[vt_idx[0]]
                    tri_uv[i, 1] = texcoords[vt_idx[1]]
                    tri_uv[i, 2] = texcoords[vt_idx[2]]
                else:
                    tri_uv[i, :] = 0.0
            # 不用纹理就保持 0

        # 把数据拷到 Taichi 的 field 里
        self.tri.from_numpy(tri)
        self.tri_ndc_z.from_numpy(tri_ndc_z)
        self.tri_world.from_numpy(tri_world)
        self.tri_uv.from_numpy(tri_uv)
        self.light_dir[None] = ti.Vector(list(light_dir))

        # 调用 kernel。即使不用纹理，也传一个很小的 dummy，保证签名一致
        if (tex_img is None) or (use_tex == 0):
            dummy = np.zeros((2, 2, 3), dtype=np.float32)
            self.draw_triangles(dummy, 0)
        else:
            self.draw_triangles(tex_img.astype(np.float32), 1)

    def get_image(self) -> np.ndarray:
        # Taichi 的字段是 (W,H,3)，我这里转成 (H,W,3) 当正常图片
        img = self.color.to_numpy()
        return np.transpose(img, (1, 0, 2))

    def save(self, path: str):
        img = self.get_image()
        ti.tools.imwrite(img, path)


# ---------------------------
# 主流程：拼一下整个渲染管线
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="一个很基础的 Taichi 软渲染器（TinyRenderer 思路）")
    parser.add_argument("--obj", type=str, required=True, help="OBJ 模型路径")
    parser.add_argument("--texture", type=str, default=None, help="可选：纹理图片路径（PNG/JPG）")
    parser.add_argument("--out", type=str, default="out.png", help="输出图片文件名")
    parser.add_argument("--width", type=int, default=800, help="图像宽")
    parser.add_argument("--height", type=int, default=800, help="图像高")
    parser.add_argument("--fov", type=float, default=60.0, help="垂直视角（度）")
    parser.add_argument("--near", type=float, default=0.1, help="近裁剪面")
    parser.add_argument("--far", type=float, default=100.0, help="远裁剪面")
    parser.add_argument("--eye", type=float, nargs=3, default=[1.0, 1.0, 3.0], help="相机位置")
    parser.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="相机看向的点")
    parser.add_argument("--up", type=float, nargs=3, default=[0.0, 1.0, 0.0], help="相机上方向")
    parser.add_argument("--bg", type=float, nargs=3, default=[0.05, 0.05, 0.08], help="背景色 (r g b)")
    parser.add_argument("--light", type=float, nargs=3, default=[0.5, 1.0, 0.3], help="方向光（世界空间）")
    args = parser.parse_args()

    # 读模型
    V, VT, F, F_vt = load_obj(args.obj)
    if V.shape[0] == 0 or F.shape[0] == 0:
        raise RuntimeError("OBJ 里没有顶点或没有面，没法画。")

    # 简单做个归一化：把模型放到原点附近，尺度大概到 1 左右
    vmin = V.min(axis=0)
    vmax = V.max(axis=0)
    center = (vmin + vmax) * 0.5
    scale = 1.0 / (np.linalg.norm(vmax - vmin) + 1e-6)
    Vn = (V - center) * (2.0 * scale)

    # 组装矩阵：View（看）+ Projection（投影）+ Viewport（视口）
    view = look_at(args.eye, args.center, args.up)
    proj = perspective(args.fov, args.width / float(args.height), args.near, args.far)
    PV = (proj @ view).astype(np.float32)                 # 先做投影*视图
    VP = viewport(args.width, args.height).astype(np.float32)  # 再把 NDC 映到屏幕

    # 创建渲染器
    renderer = TinyRendererTi(args.width, args.height, F.shape[0])

    # 清屏
    renderer.clear(args.bg[0], args.bg[1], args.bg[2])

    # 读纹理（可选）
    tex_img = None
    if args.texture is not None and os.path.exists(args.texture):
        tex_img = ti.tools.imread(args.texture).astype(np.float32) / 255.0

    # 在 CPU 端做一个简单的背面剔除（法线和相机方向点乘 > 0 就保留）
    faces_kept = []
    faces_vt_kept = []
    cam = np.array(args.eye, dtype=np.float32)
    for i in range(F.shape[0]):
        idx = F[i]
        p0, p1, p2 = Vn[idx[0]], Vn[idx[1]], Vn[idx[2]]
        n = np.cross(p1 - p0, p2 - p0)
        to_cam = cam - p0
        if np.dot(n, to_cam) > 0:
            faces_kept.append(idx)
            if F_vt.shape[0] == F.shape[0]:
                faces_vt_kept.append(F_vt[i])

    F_kept = np.array(faces_kept, dtype=np.int32)
    F_vt_kept = np.array(faces_vt_kept, dtype=np.int32) if len(faces_vt_kept) == len(faces_kept) else None
    if F_kept.shape[0] == 0:
        # 万一全被剔除了（一般很少），就退回用全部
        F_kept = F
        F_vt_kept = F_vt if (F_vt.shape[0] == F.shape[0]) else None

    # 开始渲染
    renderer.render(
        verts_world=Vn.astype(np.float32),
        faces=F_kept,
        PV=PV,
        VP=VP,
        texcoords=VT.astype(np.float32) if VT.size > 0 else None,
        faces_vt=F_vt_kept,
        light_dir=tuple(args.light),
        tex_img=tex_img,
    )

    # 存图
    renderer.save(args.out)
    print(f"[OK] Saved image to {args.out}")


if __name__ == "__main__":
    main()
