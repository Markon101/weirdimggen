use image::{ImageBuffer, RgbImage};
use noise::{NoiseFn, Perlin};
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;
use std::env;

/// A simple B-spline function for KAN-like (Kolmogorov-Arnold Network) feature extraction.
/// Instead of static linear weights, we use spline functions to map the latent chaos.
fn spline_feature_extraction(t: f64, control_points: &[f64; 4]) -> f64 {
    // Clamp t to [0, 1]
    let t = t.clamp(0.0, 1.0);
    let it = 1.0 - t;
    
    // Basis functions for cubic B-spline
    let b0 = it * it * it;
    let b1 = 3.0 * t * it * it;
    let b2 = 3.0 * t * t * it;
    let b3 = t * t * t;
    
    control_points[0] * b0 + control_points[1] * b1 + control_points[2] * b2 + control_points[3] * b3
}

/// Sigmoid activation
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// LSTM / Titans-inspired memory gate.
/// Controls how much of the new feature is stored vs how much of the old memory is kept.
fn memory_gate(x: f64, h: f64, w_x: f64, w_h: f64, b: f64) -> f64 {
    sigmoid(w_x * x + w_h * h + b)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let width = if args.len() > 1 { args[1].parse().unwrap_or(1024) } else { 1024 };
    let height = if args.len() > 2 { args[2].parse().unwrap_or(1024) } else { 1024 };
    let output_path = if args.len() > 3 { &args[3] } else { "test_output.png" };
    
    let start_time = Instant::now();
    let mut rng = rand::thread_rng();
    let seed: u32 = rng.gen();
    let perlin = Perlin::new(seed);
    
    println!("Generating image with dimensions {}x{}", width, height);
    println!("Seed: {}", seed);
    println!("Distilling latent space into photos via Titans/LSTM/KAN architecture...");
    
    // Neural architecture hyperparameters
    let layers = 6;
    let base_scale = rng.gen_range(0.002..0.008);
    
    // Spline control points for our "KAN" activation layer
    let spline_cp: [f64; 4] = [
        rng.gen_range(-1.5..1.5), 
        rng.gen_range(-1.5..1.5), 
        rng.gen_range(-1.5..1.5), 
        rng.gen_range(-1.5..1.5)
    ];

    // Multi-frequency palette parameters (from existing functionality)
    let c1: [f64; 3] = [rng.gen_range(0.5..2.0), rng.gen_range(0.5..2.0), rng.gen_range(0.5..2.0)];
    let c2: [f64; 3] = [rng.gen_range(0.1..1.0), rng.gen_range(0.1..1.0), rng.gen_range(0.1..1.0)];
    let d1: [f64; 3] = [rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)];
    let a: [f64; 3] = [rng.gen_range(0.1..0.9), rng.gen_range(0.1..0.9), rng.gen_range(0.1..0.9)];
    let b_pal: [f64; 3] = [rng.gen_range(0.2..0.5), rng.gen_range(0.2..0.5), rng.gen_range(0.2..0.5)];

    // Using Rayon for parallel processing to ensure Rust is much faster
    let mut raw_pixels = vec![0u8; (width * height * 3) as usize];
    
    // Parallelize over rows
    raw_pixels.par_chunks_mut((width * 3) as usize).enumerate().for_each(|(y, row)| {
        for x in 0..width {
            let mut c_state = 0.0;
            let mut h_state = 0.0;
            
            // "Latent space captures"
            // We iterate through layers (like time steps in LSTM or memory in Titans)
            for layer in 0..layers {
                let scale = base_scale * (layer as f64 * 1.5 + 1.0);
                
                // 1. Raw latent chaos (Perlin noise acting as latent vector z)
                let nx = x as f64 * scale;
                let ny = y as f64 * scale;
                // Add domain warping based on previous hidden state
                let warp_x = nx + h_state * 0.5;
                let warp_y = ny + h_state * 0.5;
                let chaos = perlin.get([warp_x, warp_y, layer as f64 * 1.2]);
                
                // 2. KAN-inspired Feature Extraction
                // Normalize chaos from [-1, 1] to [0, 1] for spline evaluation
                let norm_chaos = (chaos + 1.0) * 0.5;
                let feature = spline_feature_extraction(norm_chaos, &spline_cp);
                
                // 3. LSTM/Titans Memory Integration
                let prev_h = h_state;
                let prev_c = c_state;
                
                // Gates mapping feature and previous memory
                let forget = memory_gate(feature, prev_h, 0.5, 0.5, 0.1);
                let input_gate = memory_gate(feature, prev_h, 0.8, -0.2, 0.0);
                let output_gate = memory_gate(feature, prev_h, 0.3, 0.7, -0.1);
                
                // Candidate memory (analogous to C~)
                let candidate = (feature * 1.2 + prev_h * 0.4).tanh();
                
                // Update cell state and hidden state
                let next_c = forget * prev_c + input_gate * candidate;
                let next_h = output_gate * next_c.tanh();
                
                c_state = next_c;
                h_state = next_h;
            }
            
            // Map final hidden state (distilled latent feature) to visual palette
            let val = h_state;
            
            let mut rgb = [0.0; 3];
            for i in 0..3 {
                // Multi-frequency mixing for color bands
                let angle = std::f64::consts::TAU * (c1[i] * val + c2[i] * (val * 3.0).sin() + d1[i]);
                let color_val = a[i] + b_pal[i] * angle.cos();
                rgb[i] = color_val.clamp(0.0, 1.0);
            }
            
            row[(x * 3) as usize] = (rgb[0] * 255.0) as u8;
            row[(x * 3 + 1) as usize] = (rgb[1] * 255.0) as u8;
            row[(x * 3 + 2) as usize] = (rgb[2] * 255.0) as u8;
        }
    });
    
    let img: RgbImage = ImageBuffer::from_raw(width, height, raw_pixels)
        .expect("Failed to create image buffer");
    
    img.save(output_path).unwrap();
    
    let duration = start_time.elapsed();
    println!("Finished in {:.2?}s", duration);
}
