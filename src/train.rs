// Realtivistic DCGAN.
// https://github.com/AlexiaJM/RelativisticGAN
use crate::model::{self, IMG_SIZE};
use std::{error::Error, fs::create_dir_all, path::PathBuf};
use tch::{nn, nn::OptimizerConfig as _, Device, Kind, Tensor};

const BATCH_SIZE: i64 = 32;
const LEARNING_RATE: f64 = 1e-4;
const EPOCH: u64 = 100_000_000;

pub fn train(dataset: PathBuf, model: PathBuf, demo: PathBuf) -> Result<(), Box<dyn Error>> {
    let device = Device::cuda_if_available();
    let images = tch::vision::image::load_dir(dataset, IMG_SIZE, IMG_SIZE)?;
    println!("loaded dataset: {:?}", images);
    let train_size = images.size()[0];
    if !model.is_dir() {
        create_dir_all(&model)?;
    }
    if !demo.is_dir() {
        create_dir_all(&demo)?;
    }

    let random_batch_images = || {
        let index = Tensor::randint(train_size, &[BATCH_SIZE], (Kind::Int64, device));
        images
            .to_device(device)
            .index_select(0, &index)
            .to_kind(Kind::Float)
            / 127.5
            - 1.
    };

    let mut generator_vs = nn::VarStore::new(device);
    let generator = model::generator(&generator_vs.root());
    let mut opt_g = nn::adam(0.5, 0.999, 0.).build(&generator_vs, LEARNING_RATE)?;

    let mut discriminator_vs = nn::VarStore::new(device);
    let discriminator = model::discriminator(&discriminator_vs.root());
    let mut opt_d = nn::adam(0.5, 0.999, 0.).build(&discriminator_vs, LEARNING_RATE)?;

    let pb = indicatif::ProgressBar::new(EPOCH);
    let demo_noise = model::rand_latent(BATCH_SIZE, device);
    for index in 0..EPOCH {
        pb.set_position(index);
        discriminator_vs.unfreeze();
        generator_vs.freeze();
        let discriminator_loss = {
            let batch_images = random_batch_images();
            let y_pred = batch_images.apply_t(&discriminator, true);
            let y_pred_fake = model::rand_latent(BATCH_SIZE, device)
                .apply_t(&generator, true)
                .copy()
                .detach()
                .apply_t(&discriminator, true);
            model::mse_loss(&y_pred, &(y_pred_fake.mean(Kind::Float) + 1.))
                + model::mse_loss(&y_pred_fake, &(y_pred.mean(Kind::Float) - 1.))
        };
        opt_d.backward_step(&discriminator_loss);

        discriminator_vs.freeze();
        generator_vs.unfreeze();
        let generator_loss = {
            let batch_images = random_batch_images();
            let y_pred = batch_images.apply_t(&discriminator, true);
            let y_pred_fake = model::rand_latent(BATCH_SIZE, device)
                .apply_t(&generator, true)
                .apply_t(&discriminator, true);
            model::mse_loss(&y_pred, &(y_pred_fake.mean(Kind::Float) - 1))
                + model::mse_loss(&y_pred_fake, &(y_pred.mean(Kind::Float) + 1))
        };
        opt_g.backward_step(&generator_loss);
        generator_vs.freeze();

        if index % 1000 == 0 {
            let imgs = demo_noise
                .apply_t(&generator, false)
                .view([-1, 3, IMG_SIZE, IMG_SIZE])
                .to_device(Device::Cpu);
            let matrix = model::image_matrix(&imgs, 4);
            tch::vision::image::save(&matrix, demo.join(format!("demo_{}.png", index)))?;
            generator_vs.save(model.join(format!("gen_{}.pt", index)))?;
            discriminator_vs.save(model.join(format!("dis_{}.pt", index)))?;
        }
    }
    pb.finish();
    Ok(())
}
