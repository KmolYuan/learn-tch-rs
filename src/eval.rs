use crate::model::{self, IMG_SIZE};
use std::{error::Error, fs::create_dir_all, path::PathBuf};
use tch::{nn, Device};

pub fn eval(gen_path: PathBuf, demo: PathBuf) -> Result<(), Box<dyn Error>> {
    let device = Device::cuda_if_available();
    if !demo.is_dir() {
        create_dir_all(&demo)?;
    }

    let mut generator_vs = nn::VarStore::new(device);
    let generator = model::generator(&generator_vs.root());
    generator_vs.load(gen_path)?;
    generator_vs.freeze();
    let imgs = model::rand_latent(1, device)
        .apply_t(&generator, false)
        .view([-1, 3, IMG_SIZE, IMG_SIZE])
        .to_device(Device::Cpu);
    let matrix = model::image_matrix(&imgs, 1);
    tch::vision::image::save(&matrix, demo.join("eval.png"))?;
    Ok(())
}
