use image::{ImageBuffer, RgbImage};
use noise::{NoiseFn, Perlin};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Width of the image
    #[arg(long, default_value_t = 1024)]
    width: u32,

    /// Height of the image
    #[arg(long, default_value_t = 1024)]
    height: u32,

    /// Output path for the image
    #[arg(short, long, default_value = "/sdcard/noise.png")]
    out: String,

    /// Number of images to generate
    #[arg(short, long, default_value_t = 1)]
    count: i32,

    /// Path to the weights file
    #[arg(long, default_value = "weights.json")]
    weights: String,

    /// Force new random weights even if a weights file exists
    #[arg(long)]
    new_weights: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Weights {
    seed: u32,
    layers: usize,
    base_scale: f64,
    spline_cp: [f64; 4],
    c1: [f64; 3],
    c2: [f64; 3],
    d1: [f64; 3],
    a: [f64; 3],
    b_pal: [f64; 3],
    // LSTM weight approximations
    w_x: f64,
    w_h: f64,
    b_gate: f64,
}

impl Weights {
    fn random() -> Self {
        let mut rng = rand::thread_rng();
        Weights {
            seed: rng.gen(),
            layers: 6,
            base_scale: rng.gen_range(0.002..0.008),
            spline_cp: [
                rng.gen_range(-1.5..1.5),
                rng.gen_range(-1.5..1.5),
                rng.gen_range(-1.5..1.5),
                rng.gen_range(-1.5..1.5),
            ],
            c1: [rng.gen_range(0.5..2.0), rng.gen_range(0.5..2.0), rng.gen_range(0.5..2.0)],
            c2: [rng.gen_range(0.1..1.0), rng.gen_range(0.1..1.0), rng.gen_range(0.1..1.0)],
            d1: [rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)],
            a: [rng.gen_range(0.1..0.9), rng.gen_range(0.1..0.9), rng.gen_range(0.1..0.9)],
            b_pal: [rng.gen_range(0.2..0.5), rng.gen_range(0.2..0.5), rng.gen_range(0.2..0.5)],
            w_x: rng.gen_range(0.3..0.9),
            w_h: rng.gen_range(0.3..0.9),
            b_gate: rng.gen_range(-0.2..0.2),
        }
    }

    fn load(path: &str) -> Option<Self> {
        let mut file = File::open(path).ok()?;
        let mut contents = String::new();
        file.read_to_string(&mut contents).ok()?;
        serde_json::from_str(&contents).ok()
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let serialized = serde_json::to_string_pretty(self).unwrap();
        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())
    }
}

fn spline_feature_extraction(t: f64, control_points: &[f64; 4]) -> f64 {
    let t = t.clamp(0.0, 1.0);
    let it = 1.0 - t;
    let b0 = it * it * it;
    let b1 = 3.0 * t * it * it;
    let b2 = 3.0 * t * t * it;
    let b3 = t * t * t;
    control_points[0] * b0 + control_points[1] * b1 + control_points[2] * b2 + control_points[3] * b3
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn memory_gate(x: f64, h: f64, w_x: f64, w_h: f64, b: f64) -> f64 {
    sigmoid(w_x * x + w_h * h + b)
}

fn generate_image(args: &Args, weights: &Weights, output_path: &str) {
    let perlin = Perlin::new(weights.seed);
    let mut raw_pixels = vec![0u8; (args.width * args.height * 3) as usize];

    raw_pixels.par_chunks_mut((args.width * 3) as usize).enumerate().for_each(|(y, row)| {
        for x in 0..args.width {
            let mut c_state = 0.0;
            let mut h_state = 0.0;
            
            for layer in 0..weights.layers {
                let scale = weights.base_scale * (layer as f64 * 1.5 + 1.0);
                let nx = x as f64 * scale;
                let ny = y as f64 * scale;
                let warp_x = nx + h_state * 0.5;
                let warp_y = ny + h_state * 0.5;
                let chaos = perlin.get([warp_x, warp_y, layer as f64 * 1.2]);
                
                let norm_chaos = (chaos + 1.0) * 0.5;
                let feature = spline_feature_extraction(norm_chaos, &weights.spline_cp);
                
                let prev_h = h_state;
                let prev_c = c_state;
                
                let forget = memory_gate(feature, prev_h, weights.w_x, weights.w_h, weights.b_gate + 0.1);
                let input_gate = memory_gate(feature, prev_h, weights.w_x + 0.3, weights.w_h - 0.7, weights.b_gate);
                let output_gate = memory_gate(feature, prev_h, weights.w_x - 0.5, weights.w_h + 0.2, weights.b_gate - 0.1);
                
                let candidate = (feature * 1.2 + prev_h * 0.4).tanh();
                let next_c = forget * prev_c + input_gate * candidate;
                let next_h = output_gate * next_c.tanh();
                
                c_state = next_c;
                h_state = next_h;
            }
            
            let val = h_state;
            let mut rgb = [0.0; 3];
            for i in 0..3 {
                let angle = std::f64::consts::TAU * (weights.c1[i] * val + weights.c2[i] * (val * 3.0).sin() + weights.d1[i]);
                let color_val = weights.a[i] + weights.b_pal[i] * angle.cos();
                rgb[i] = color_val.clamp(0.0, 1.0);
            }
            
            row[(x * 3) as usize] = (rgb[0] * 255.0) as u8;
            row[(x * 3 + 1) as usize] = (rgb[1] * 255.0) as u8;
            row[(x * 3 + 2) as usize] = (rgb[2] * 255.0) as u8;
        }
    });
    
    let img: RgbImage = ImageBuffer::from_raw(args.width, args.height, raw_pixels)
        .expect("Failed to create image buffer");
    img.save(output_path).unwrap();
}

fn main() {
    let args = Args::parse();

    // Optimize for 6 threads on Snapdragon
    rayon::ThreadPoolBuilder::new().num_threads(6).build_global().unwrap();

    let weights = if args.new_weights || !Path::new(&args.weights).exists() {
        println!("Generating new random weights...");
        let w = Weights::random();
        if let Err(e) = w.save(&args.weights) {
            eprintln!("Warning: Could not save weights: {}", e);
        }
        w
    } else {
        println!("Loading weights from {}...", args.weights);
        Weights::load(&args.weights).unwrap_or_else(|| {
            println!("Failed to load weights, using random instead.");
            Weights::random()
        })
    };

    println!("Generating {} image(s) with dimensions {}x{}", args.count, args.width, args.height);
    println!("Weights Seed: {}", weights.seed);
    
    for i in 0..args.count {
        let start_time = Instant::now();
        let mut current_out = args.out.clone();
        if args.count > 1 {
            if current_out.contains(".png") {
                current_out = current_out.replace(".png", &format!("_{}.png", i));
            } else {
                current_out = format!("{}_{}.png", current_out, i);
            }
        }

        println!("[{}/{}] Distilling to '{}'...", i + 1, args.count, current_out);
        generate_image(&args, &weights, &current_out);
        
        let duration = start_time.elapsed();
        println!("Finished in {:.2?}s", duration);
    }
}
