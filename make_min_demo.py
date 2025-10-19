# make_min_demo.py
# This script creates two minimal OBJ files (no texture) for verifying the renderer.
# 1) single_triangle.obj: one triangle centered in view
# 2) uv_square.obj: a square made of two triangles
# No external dependencies required.

import os

triangle_obj = r"""
# Single triangle in XY plane (Z=0)
v  -0.8 -0.6  0.0
v   0.8 -0.6  0.0
v   0.0  0.7  0.0
f 1 2 3
""".strip() + "\n"

square_obj = r"""
# Unit-like square in XY plane (Z=0), composed of two triangles
v -1.0 -1.0 0.0
v  1.0 -1.0 0.0
v  1.0  1.0 0.0
v -1.0  1.0 0.0
f 1 2 3
f 1 3 4
""".strip() + "\n"

with open("single_triangle.obj", "w", encoding="utf-8") as f:
    f.write(triangle_obj)
with open("uv_square.obj", "w", encoding="utf-8") as f:
    f.write(square_obj)

print("[OK] Created:")
print(" - single_triangle.obj")
print(" - uv_square.obj")

print("\nRun these commands to render (no texture needed):\n")
print("python taichi_tinyrenderer.py --obj single_triangle.obj --out triangle_out.png "
      "--width 512 --height 512 --eye 0 0 3 --center 0 0 0 --up 0 1 0")
print("python taichi_tinyrenderer.py --obj uv_square.obj --out square_out.png "
      "--width 512 --height 512 --eye 0 0 3 --center 0 0 0 --up 0 1 0")
