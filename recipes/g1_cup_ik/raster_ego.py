"""无 Kit RTX 时：把杯/桌/腕投影成 480x640 ego_view，好让 GR00T 仍能训。

Isaac Sim 5.1 + Blackwell 上 ``--enable_cameras`` 会在 hydra 里崩，所以默认走这套
几何光栅。杯位姿域随机会进画面；真机 ego 有域差，Kit 相机修好后把 ENABLE_CAMERAS=1。
"""

from __future__ import annotations

import torch

EGO_H = 480
EGO_W = 640


def _look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """eye/target/up: (N,3) → cam-from-world rotation (N,3,3), rows = cam axes."""
    z = target - eye
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x = torch.cross(z, up, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    y = torch.cross(x, z, dim=-1)
    return torch.stack([x, y, z], dim=1)


def raster_ego_view(
    cup_pos: torch.Tensor,
    cup_radius: torch.Tensor,
    cup_height: torch.Tensor,
    table_z: torch.Tensor,
    wrist_pos: torch.Tensor,
    *,
    height: int = EGO_H,
    width: int = EGO_W,
) -> torch.Tensor:
    """cup/wrist 在 env 系 (N,3)；返回 uint8 RGB (N,H,W,3)。"""
    n, device = cup_pos.shape[0], cup_pos.device
    eye = torch.tensor([0.04, -0.10, 1.18], device=device).expand(n, 3)
    target = torch.stack(
        [
            cup_pos[:, 0] * 0.35 + 0.12,
            cup_pos[:, 1] * 0.35 + 0.22,
            table_z + 0.06,
        ],
        dim=-1,
    )
    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    r_cw = _look_at(eye, target, up)
    fov = 1.05  # ~60 deg
    fx = 0.5 * width / torch.tan(torch.tensor(fov * 0.5, device=device))
    fy = fx
    cx, cy = width * 0.5, height * 0.5

    img = torch.zeros(n, height, width, 3, device=device, dtype=torch.uint8)
    img[..., 0] = 38
    img[..., 1] = 40
    img[..., 2] = 48

    def project(p_w: torch.Tensor):
        p_c = torch.bmm(r_cw, (p_w - eye).unsqueeze(-1)).squeeze(-1)
        z = p_c[:, 2].clamp_min(1e-3)
        u = fx * p_c[:, 0] / z + cx
        v = height - (fy * p_c[:, 1] / z + cy)
        return u, v, z

    # 桌面四角
    tw, td = 0.35, 0.275
    corners = torch.stack(
        [
            torch.stack([torch.full((n,), 0.18 - tw, device=device), torch.full((n,), 0.36 - td, device=device), table_z], -1),
            torch.stack([torch.full((n,), 0.18 + tw, device=device), torch.full((n,), 0.36 - td, device=device), table_z], -1),
            torch.stack([torch.full((n,), 0.18 + tw, device=device), torch.full((n,), 0.36 + td, device=device), table_z], -1),
            torch.stack([torch.full((n,), 0.18 - tw, device=device), torch.full((n,), 0.36 + td, device=device), table_z], -1),
        ],
        dim=1,
    )
    us, vs = [], []
    for k in range(4):
        u, v, _ = project(corners[:, k])
        us.append(u)
        vs.append(v)
    u_min = torch.stack(us, 1).min(1).values.clamp(0, width - 1).long()
    u_max = torch.stack(us, 1).max(1).values.clamp(0, width - 1).long()
    v_min = torch.stack(vs, 1).min(1).values.clamp(0, height - 1).long()
    v_max = torch.stack(vs, 1).max(1).values.clamp(0, height - 1).long()
    yy = torch.arange(height, device=device).view(1, height, 1)
    xx = torch.arange(width, device=device).view(1, 1, width)
    table_mask = (
        (xx >= u_min.view(n, 1, 1))
        & (xx <= u_max.view(n, 1, 1))
        & (yy >= v_min.view(n, 1, 1))
        & (yy <= v_max.view(n, 1, 1))
    )
    img[table_mask] = torch.tensor([92, 72, 48], device=device, dtype=torch.uint8)

    # 杯：投影中心 + 半径
    cup_top = cup_pos.clone()
    cup_top[:, 2] = cup_pos[:, 2] + cup_height * 0.5
    u, v, z = project(cup_top)
    pix_r = (fx * cup_radius / z).clamp(3.0, 80.0)
    cu = u.round().long()
    cv = v.round().long()
    rr = pix_r.round().long().clamp(min=2)
    dist2 = (xx - cu.view(n, 1, 1)) ** 2 + (yy - cv.view(n, 1, 1)) ** 2
    cup_mask = dist2 <= (rr.view(n, 1, 1) ** 2)
    img[cup_mask] = torch.tensor([40, 120, 200], device=device, dtype=torch.uint8)

    u, v, z = project(wrist_pos)
    pix_r = (fx * 0.035 / z.clamp_min(0.05)).clamp(2.0, 40.0)
    cu = u.round().long()
    cv = v.round().long()
    rr = pix_r.round().long().clamp(min=2)
    dist2 = (xx - cu.view(n, 1, 1)) ** 2 + (yy - cv.view(n, 1, 1)) ** 2
    wrist_mask = dist2 <= (rr.view(n, 1, 1) ** 2)
    img[wrist_mask] = torch.tensor([40, 200, 90], device=device, dtype=torch.uint8)
    return img
