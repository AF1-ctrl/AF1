#!/usr/bin/env python3
"""
Banana Time - procedural textures for the collectible items (owner: items).

Run with the system Python (needs numpy, PIL, scipy - Blender's bundled Python has no PIL):
    python3 tools/blender/items_textures.py

Writes (Assets/BananaTime/Art/Textures):
  T_Coconut.png    1024x512 sRGB  fibrous hairy husk, equirect for the coconut UV sphere,
                   fibres run along v (pole to pole = meridians). Seamless in u AND v.
  T_Coconut_N.png  1024x512       tangent-space normal map (OpenGL / Unity convention, +Y = +v)
                   derived from the same height field. Seamless in u AND v.

Deterministic (fixed seeds) so re-running reproduces identical files.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

OUT = "/home/user/AF1/Assets/BananaTime/Art/Textures"
W, H = 1024, 512


def periodic_noise(rng, w, h, fx, fy, power=1.0):
    """Band-limited periodic noise via FFT. fx / fy = cut-off in cycles per image in x / y."""
    n = rng.standard_normal((h, w))
    F = np.fft.rfft2(n)
    ky = np.fft.fftfreq(h) * h            # cycles per image
    kx = np.fft.rfftfreq(w) * w
    KX, KY = np.meshgrid(kx, ky)
    G = np.exp(-(KX / fx) ** 2 - (KY / fy) ** 2) ** power
    G[0, 0] = 0
    out = np.fft.irfft2(F * G, s=(h, w))
    out -= out.mean()
    out /= (out.std() + 1e-9)
    return out


def wrap_warp(img, dx, dy):
    """Resample img at (y+dy, x+dx) with wrap-around (keeps the texture seamless)."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return ndimage.map_coordinates(img, [yy + dy, xx + dx], order=1, mode='grid-wrap')


def draw_strands(rng, w, h, count, length, width, angle_sd, value, ss=2):
    """Thin, slightly curved hair strands running mostly along v; drawn wrapped for seamlessness.
    Returns float array 0..1 (coverage) and a signed height contribution."""
    S = ss
    img = Image.new("L", (w * S, h * S), 0)
    d = ImageDraw.Draw(img)
    for _ in range(count):
        x0 = rng.uniform(0, w)
        y0 = rng.uniform(0, h)
        L = rng.uniform(0.5, 1.0) * length
        ang = np.radians(90 + rng.normal(0, angle_sd))    # mostly vertical
        bend = rng.normal(0, 0.25)
        pts = []
        steps = 10
        for i in range(steps + 1):
            t = i / steps - 0.5
            a = ang + bend * t
            pts.append((x0 + np.cos(a) * L * t, y0 + np.sin(a) * L * t))
        lw = max(1, int(round(rng.uniform(0.6, 1.4) * width * S)))
        val = int(255 * rng.uniform(0.55, 1.0) * value)
        for ox in (-w, 0, w):
            for oy in (-h, 0, h):
                d.line([((px + ox) * S, (py + oy) * S) for px, py in pts], fill=val, width=lw)
    img = img.resize((w, h), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def lerp_palette(t, stops):
    """t: array 0..1 ; stops: list of (pos, (r,g,b)) sRGB 0..1."""
    t = np.clip(t, 0, 1)
    out = np.zeros(t.shape + (3,))
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops])
    for c in range(3):
        out[..., c] = np.interp(t, pos, cols[:, c])
    return out


def height_to_normal(hgt, strength):
    """Tangent-space normal map. u = +x (columns), v = +y upward (rows go DOWN in the image)."""
    dhdu = (np.roll(hgt, -1, axis=1) - np.roll(hgt, 1, axis=1)) * 0.5
    dhdv = (np.roll(hgt, 1, axis=0) - np.roll(hgt, -1, axis=0)) * 0.5   # row-1 is +v
    nx = -dhdu * strength
    ny = -dhdv * strength
    nz = np.ones_like(hgt)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / l, ny / l, nz / l], axis=-1)
    return ((n * 0.5 + 0.5) * 255 + 0.5).clip(0, 255).astype(np.uint8)


def coconut():
    rng = np.random.default_rng(20260923)
    # --- fibre field: fine across u, long along v -----------------------------------------
    fib1 = periodic_noise(rng, W, H, fx=150, fy=7)        # fine fibres
    fib2 = periodic_noise(rng, W, H, fx=55, fy=4)         # fibre bundles
    fib3 = periodic_noise(rng, W, H, fx=320, fy=14)       # very fine grain
    # gentle waviness: displace x by a smooth periodic field (fibres meander a little)
    wx = periodic_noise(rng, W, H, fx=4, fy=3) * 4.0
    wx2 = periodic_noise(rng, W, H, fx=10, fy=6) * 1.2
    fib = 0.55 * fib1 + 0.40 * fib2 + 0.22 * fib3
    fib = wrap_warp(fib, wx + wx2, np.zeros_like(wx))
    # sharpen fibres into ridges
    fib = np.tanh(fib * 0.9)

    # --- hair strands (lighter, loose) + dark gaps ----------------------------------------
    hair_l = draw_strands(rng, W, H, count=650, length=80, width=1.1, angle_sd=11, value=1.0)
    hair_d = draw_strands(rng, W, H, count=450, length=60, width=1.3, angle_sd=8, value=1.0)
    hair_l = ndimage.gaussian_filter(hair_l, 0.5, mode='wrap')

    # --- husk plates: periodic Voronoi grooves (irregular panels like the concept coconut) --
    cells = [(rng.uniform(0, W), rng.uniform(0, H)) for _ in range(22)]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    xx = xx + wx * 1.5
    d1 = np.full((H, W), 1e9); d2 = np.full((H, W), 1e9)
    for cx, cy in cells:
        dx = np.abs(xx - cx); dx = np.minimum(dx, W - dx)
        dy = np.abs(yy - cy); dy = np.minimum(dy, H - dy)
        dd = np.sqrt(dx * dx + (dy * 1.35) ** 2)
        m = dd < d1
        d2 = np.where(m, d1, np.minimum(d2, dd))
        d1 = np.where(m, dd, d1)
    edge = d2 - d1
    cracks = np.exp(-(edge / 5.0) ** 2)
    cracks *= 0.6 + 0.4 * (periodic_noise(rng, W, H, fx=8, fy=6) > -0.3)
    cracks = ndimage.gaussian_filter(cracks, 1.0, mode='wrap')
    # plates slightly domed: brighter in the middle of each panel
    plate = np.clip(edge / 40.0, 0, 1)

    # --- macro variation ---------------------------------------------------------------------
    macro = periodic_noise(rng, W, H, fx=4, fy=3)
    blot = periodic_noise(rng, W, H, fx=16, fy=10)

    # height (0..1) used for colour + normal map
    hgt = 0.44 + 0.26 * fib + 0.28 * hair_l - 0.22 * hair_d - 0.38 * cracks + 0.12 * plate
    hgt = np.clip(hgt, 0, 1)

    tone = hgt + 0.06 * macro + 0.04 * blot
    stops = [
        (0.00, (0.16, 0.075, 0.03)),
        (0.25, (0.30, 0.15, 0.06)),
        (0.50, (0.47, 0.26, 0.11)),
        (0.72, (0.62, 0.37, 0.17)),
        (0.90, (0.78, 0.52, 0.28)),
        (1.00, (0.86, 0.64, 0.38)),
    ]
    rgb = lerp_palette(tone, stops)
    # slight warm/cool variation in the plates
    rgb[..., 0] *= 1.0 + 0.04 * blot
    rgb[..., 2] *= 1.0 - 0.06 * blot
    rgb = np.clip(rgb, 0, 1)
    Image.fromarray((rgb * 255 + 0.5).astype(np.uint8), "RGB").save(os.path.join(OUT, "T_Coconut.png"))

    # normal map from a slightly smoothed height (avoid aliasing sparkle on mobile)
    hn = ndimage.gaussian_filter(hgt, 0.7, mode='wrap')
    nrm = height_to_normal(hn, strength=6.0)
    Image.fromarray(nrm, "RGB").save(os.path.join(OUT, "T_Coconut_N.png"))
    print("wrote T_Coconut.png, T_Coconut_N.png", rgb.mean(axis=(0, 1)))


def check_seamless(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    du = np.abs(a[:, 0] - a[:, -1]).mean()
    dv = np.abs(a[0] - a[-1]).mean()
    inner = np.abs(a[:, 1:] - a[:, :-1]).mean()
    print(f"{os.path.basename(path)}: edge diff u={du:.2f} v={dv:.2f} (typical neighbour diff {inner:.2f})")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    coconut()
    for n in ("T_Coconut.png", "T_Coconut_N.png"):
        check_seamless(os.path.join(OUT, n))
