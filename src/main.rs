use image::{ImageBuffer, RgbImage};
use noise::{NoiseFn, Perlin};
use rand::Rng;
use std::path::Path;
use std::time::Instant;
use clap::Parser;

use candle_core::{Device, Tensor, DType, Result as CandleResult, Module};
use candle_nn::{Linear, linear, VarBuilder, Optimizer, AdamW, ParamsAdamW, VarMap, Init};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 1024)]
    width: u32,

    #[arg(long, default_value_t = 1024)]
    height: u32,

    #[arg(short, long, default_value = "test_output.png")]
    out: String,

    #[arg(short, long, default_value_t = 1)]
    count: i32,

    #[arg(long, default_value = "model.safetensors")]
    weights: String,

    #[arg(long)]
    new_weights: bool,
}

// SIREN-like Coordinate Network
struct SirenLayer {
    linear: Linear,
    w0: f32,
}

impl SirenLayer {
    fn new(in_dim: usize, out_dim: usize, w0: f32, is_first: bool, vb: VarBuilder) -> CandleResult<Self> {
        // Standard SIREN initialization
        let bound = if is_first {
            1.0 / in_dim as f64
        } else {
            (6.0 / in_dim as f64).sqrt() / w0 as f64
        };
        
        let weight = vb.get_with_hints(
            (out_dim, in_dim),
            "weight",
            Init::Uniform { lo: -bound, up: bound },
        )?;
        let bias = vb.get_with_hints(
            out_dim,
            "bias",
            Init::Uniform { lo: -bound, up: bound },
        )?;
        
        let linear = Linear::new(weight, Some(bias));
        Ok(Self { linear, w0 })
    }
}

impl Module for SirenLayer {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let xs = self.linear.forward(xs)?;
        (xs * self.w0 as f64)?.sin()
    }
}

struct EvolvingNet {
    layer1: SirenLayer,
    layer2: SirenLayer,
    layer3: SirenLayer,
    layer4: SirenLayer,
    out_layer: Linear,
}

impl EvolvingNet {
    fn new(vb: VarBuilder) -> CandleResult<Self> {
        let hidden = 128; 
        let input_dim = 3; // nx, ny, r
        let w0 = 30.0;
        
        let layer1 = SirenLayer::new(input_dim, hidden, w0, true, vb.pp("l1"))?;
        let layer2 = SirenLayer::new(hidden, hidden, w0, false, vb.pp("l2"))?;
        let layer3 = SirenLayer::new(hidden, hidden, w0, false, vb.pp("l3"))?;
        let layer4 = SirenLayer::new(hidden, hidden, w0, false, vb.pp("l4"))?;
        let out_layer = linear(hidden, 3, vb.pp("out"))?;
        Ok(Self { layer1, layer2, layer3, layer4, out_layer })
    }

    fn forward_batch(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let xs = self.layer1.forward(xs)?;
        let xs = self.layer2.forward(&xs)?;
        let xs = self.layer3.forward(&xs)?;
        let xs = self.layer4.forward(&xs)?;
        let xs = self.out_layer.forward(&xs)?;
        candle_nn::ops::sigmoid(&xs)
    }
}

impl Module for EvolvingNet {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.forward_batch(xs)
    }
}

fn generate_image(args: &Args, net: &EvolvingNet, device: &Device, output_path: &str) -> CandleResult<()> {
    let mut raw_pixels = vec![0u8; (args.width * args.height * 3) as usize];
    
    println!("  Rendering {}x{}...", args.width, args.height);
    
    for y in 0..args.height {
        let mut xs_vec = Vec::with_capacity(args.width as usize * 3);
        let ny = (y as f32 / args.height as f32) * 2.0 - 1.0;
        
        for x in 0..args.width {
            let nx = (x as f32 / args.width as f32) * 2.0 - 1.0;
            let r = (nx * nx + ny * ny).sqrt();
            xs_vec.push(nx);
            xs_vec.push(ny);
            xs_vec.push(r);
        }
        
        let xs_tensor = Tensor::from_vec(xs_vec, (args.width as usize, 3), device)?;
        let out_tensor = net.forward(&xs_tensor)?;
        let out_data = out_tensor.to_vec2::<f32>()?;
        
        for x in 0..args.width as usize {
            let r = (out_data[x][0].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (out_data[x][1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (out_data[x][2].clamp(0.0, 1.0) * 255.0) as u8;
            
            let idx = ((y as usize * args.width as usize) + x) * 3;
            raw_pixels[idx] = r;
            raw_pixels[idx + 1] = g;
            raw_pixels[idx + 2] = b;
        }
    }
    
    let img: RgbImage = ImageBuffer::from_raw(args.width, args.height, raw_pixels)
        .expect("Failed to create image buffer");
    img.save(output_path).unwrap();
    
    Ok(())
}

fn fractal_noise(perlin: &Perlin, x: f64, y: f64, z: f64, octaves: i32) -> f64 {
    let mut val = 0.0;
    let mut freq = 1.0;
    let mut amp = 1.0;
    let mut max_val = 0.0;
    for _ in 0..octaves {
        val += perlin.get([x * freq, y * freq, z]) * amp;
        max_val += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    (val / max_val + 1.0) * 0.5
}

fn online_learning_step(varmap: &mut VarMap, net: &EvolvingNet, device: &Device) -> CandleResult<()> {
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW {
        lr: 2e-4, // Increased learning rate slightly
        ..Default::default()
    })?;
    let mut rng = rand::thread_rng();
    let perlin = Perlin::new(rng.gen());
    
    let batch_size = 8192; // Increased batch size for more stable gradients
    let steps = 300;      // More steps to capture high frequency details
    
    let target_z: f64 = rng.gen_range(-100.0..100.0);
    let noise_scale: f64 = rng.gen_range(20.0..50.0); // Much higher scale for sharper, smaller features
    
    println!("Online Learning Phase: Adapting to complex high-frequency fractal targets...");
    
    for step in 0..steps {
        let mut xs_vec = Vec::with_capacity(batch_size * 3);
        let mut targets_vec = Vec::with_capacity(batch_size * 3);
        
        for _ in 0..batch_size {
            let nx: f32 = rng.gen_range(-1.0..1.0);
            let ny: f32 = rng.gen_range(-1.0..1.0);
            let r = (nx * nx + ny * ny).sqrt();
            
            xs_vec.push(nx);
            xs_vec.push(ny);
            xs_vec.push(r);
            
            let t1 = fractal_noise(&perlin, nx as f64 * noise_scale, ny as f64 * noise_scale, target_z, 6);
            let t2 = fractal_noise(&perlin, nx as f64 * noise_scale, ny as f64 * noise_scale, target_z + 20.0, 4);
            let t3 = fractal_noise(&perlin, nx as f64 * noise_scale, ny as f64 * noise_scale, target_z - 20.0, 2);
            
            targets_vec.push(t1 as f32);
            targets_vec.push(t2 as f32);
            targets_vec.push(t3 as f32);
        }
        
        let xs = Tensor::from_vec(xs_vec, (batch_size, 3), device)?;
        let targets = Tensor::from_vec(targets_vec, (batch_size, 3), device)?;
        
        let preds = net.forward(&xs)?;
        let loss = candle_nn::loss::mse(&preds, &targets)?;
        
        opt.backward_step(&loss)?;
        
        if step % 20 == 0 || step == steps - 1 {
            let loss_val = loss.to_vec0::<f32>()?;
            println!("  Step {}/{} - Loss: {:.6}", step, steps, loss_val);
        }
    }
    
    Ok(())
}

fn main() -> CandleResult<()> {
    let args = Args::parse();
    
    rayon::ThreadPoolBuilder::new().num_threads(8).build_global().unwrap();
    let device = Device::Cpu;

    let mut varmap = VarMap::new();
    let path = Path::new(&args.weights);
    
    if !args.new_weights && path.exists() {
        println!("Attempting to load weights from {}...", args.weights);
        if let Err(e) = varmap.load(path) {
            println!("Could not load weights: {}. Initializing new.", e);
        }
    } else {
        println!("Initializing new network weights...");
    }

    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let net = EvolvingNet::new(vb)?;

    println!("Generating {} image(s) with dimensions {}x{}", args.count, args.width, args.height);
    
    for i in 0..args.count {
        let start_time = Instant::now();
        online_learning_step(&mut varmap, &net, &device)?;
        varmap.save(&args.weights)?;
        
        let mut current_out = args.out.clone();
        if args.count > 1 {
            if current_out.contains(".png") {
                current_out = current_out.replace(".png", &format!("_{}.png", i));
            } else {
                current_out = format!("{}_{}.png", current_out, i);
            }
        }
        println!("[{}/{}] Rendering to '{}'...", i + 1, args.count, current_out);
        generate_image(&args, &net, &device, &current_out)?;
        
        let duration = start_time.elapsed();
        println!("Finished in {:.2?}s", duration);
    }
    
    Ok(())
}
