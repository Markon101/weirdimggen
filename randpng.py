#!/usr/bin/env python3
"""
Generates high-interest abstract images using Fractal Integrals (iterative vector 
field integration), recursive domain warping, and multi-frequency color palettes.
"""

import argparse
import numpy as np
from PIL import Image
import time
import random

def generate_noise_layer(shape, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    """Vectorized fractal noise generation."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    rng = np.random.default_rng(seed)
    ny, nx = shape
    noise = np.zeros(shape)
    
    frequency = 1.0 / scale
    amplitude = 1.0
    
    for _ in range(octaves):
        grid_h = int(ny * frequency) + 2
        grid_w = int(nx * frequency) + 2
        gx = rng.standard_normal((grid_h, grid_w))
        gy = rng.standard_normal((grid_h, grid_w))
        
        y, x = np.mgrid[0:ny, 0:nx]
        y_scaled = y * frequency
        x_scaled = x * frequency
        
        iy = y_scaled.astype(int)
        ix = x_scaled.astype(int)
        
        fy = y_scaled - iy
        fx = x_scaled - ix
        
        wy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
        wx = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
        
        g00 = gx[iy, ix] * fx + gy[iy, ix] * fy
        g10 = gx[iy, ix+1] * (fx - 1) + gy[iy, ix+1] * fy
        g01 = gx[iy+1, ix] * fx + gy[iy+1, ix] * (fy - 1)
        g11 = gx[iy+1, ix+1] * (fx - 1) + gy[iy+1, ix+1] * (fy - 1)
        
        n0 = g00 * (1 - wx) + g10 * wx
        n1 = g01 * (1 - wx) + g11 * wx
        layer = n0 * (1 - wy) + n1 * wy
        
        noise += layer * amplitude
        amplitude *= persistence
        frequency *= lacunarity
        
    return noise

def get_dynamic_palette():
    """Generates a complex multi-frequency trigonometric palette."""
    # a + b * cos(2pi * (c*t + d))
    # We use multiple layers of 'c' and 'd' for more "integrated" color depth
    a = np.array([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)])
    b = np.array([random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)])
    
    # Randomize frequencies and phases
    c1 = np.array([random.uniform(0.5, 2.0), random.uniform(0.5, 2.0), random.uniform(0.5, 2.0)])
    c2 = np.array([random.uniform(0.1, 1.0), random.uniform(0.1, 1.0), random.uniform(0.1, 1.0)])
    d1 = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
    
    def palette_func(v):
        v_stack = v[:, :, np.newaxis]
        # Multi-frequency mixing for more color "bands" and variety
        angle = 2 * np.pi * (c1 * v_stack + c2 * np.sin(v_stack * 3.0) + d1)
        res = a + b * np.cos(angle)
        return (np.clip(res, 0, 1) * 255).astype(np.uint8)
    
    return palette_func

def fractal_integral_composition(width, height, seed):
    """
    Implements a Fractal Integral effect by accumulating noise gradients.
    This simulates the path a particle would take through a noise field.
    """
    print("Integrating fractal field...")
    iterations = random.randint(3, 5)
    f = np.zeros((height, width))
    
    # We'll maintain a cumulative displacement field (the "integral")
    dx_total = np.zeros((height, width))
    dy_total = np.zeros((height, width))
    
    base_scale = random.uniform(250, 500)
    
    for i in range(iterations):
        # Each step, we generate noise at a different frequency
        current_scale = base_scale / (1.5 ** i)
        n = generate_noise_layer((height, width), scale=current_scale, octaves=4, seed=seed + i)
        
        # Calculate the gradient of this noise layer
        gy, gx = np.gradient(n)
        
        # Add to the total "integral" displacement
        # This creates the characteristic "flow" look of fractal integrals
        dx_total += gx * (100.0 / (i + 1))
        dy_total += gy * (100.0 / (i + 1))
        
        # Blend the noise layer into the final result, influenced by current displacement
        f += n * (0.8 ** i)
        
    # Final domain warp using the accumulated "integral" displacement
    # Mix the accumulated gradients back into the noise
    f = f + 0.5 * (np.sin(dx_total * 0.05) + np.cos(dy_total * 0.05))
    
    return f

def generate_complex_image(width, height):
    seed = random.randint(0, 2**32 - 1)
    print(f"Seed: {seed}")
    
    # 1. Fractal Integral Step
    f = fractal_integral_composition(width, height, seed)
    
    # 2. Normalize and Post-Process
    f = (f - f.min()) / (f.max() - f.min())
    
    # Optional "ridged" transformation for more structural detail
    if random.random() > 0.4:
        f = 1.0 - np.abs(f * 2.0 - 1.0)
        f = np.power(f, 1.5)
        
    # 3. Dynamic Palette
    print("Applying multi-frequency palette...")
    palette = get_dynamic_palette()
    rgb_data = palette(f)
    
    return Image.fromarray(rgb_data, 'RGB')

def main():
    parser = argparse.ArgumentParser(description="Fractal Integral Art Generator")
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--out', default='/sdcard/noise.png', help="Output path")
    parser.add_argument('-n', '--count', type=int, default=1, help="Number of images")
    args = parser.parse_args()

    for i in range(args.count):
        start_time = time.time()
        out_path = args.out
        if args.count > 1:
            out_path = out_path.replace('.png', f'_{i}.png') if '.png' in out_path else f"{out_path}_{i}.png"

        print(f"[{i+1}/{args.count}] Generating '{out_path}'...")
        img = generate_complex_image(args.width, args.height)
        img.save(out_path)
        
        duration = time.time() - start_time
        print(f"Finished in {duration:.2f}s")

if __name__ == '__main__':
    main()
