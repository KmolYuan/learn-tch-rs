// Realtivistic DCGAN.
// https://github.com/AlexiaJM/RelativisticGAN
//
// TODO: override the initializations if this does not converge well.
use std::error::Error;
use tch::{nn, nn::OptimizerConfig as _, Device, Kind, Tensor};

const IMG_SIZE: i64 = 64;
const LATENT_DIM: i64 = 128;
const BATCH_SIZE: i64 = 32;
const LEARNING_RATE: f64 = 1e-4;
const BATCHES: u64 = 100_000_000;

fn tr2d(p: nn::Path, c_in: i64, c_out: i64, padding: i64, stride: i64) -> nn::ConvTranspose2D {
    let cfg = nn::ConvTransposeConfig {
        stride,
        padding,
        bias: false,
        ..Default::default()
    };
    nn::conv_transpose2d(p, c_in, c_out, 4, cfg)
}

fn conv2d(p: nn::Path, c_in: i64, c_out: i64, padding: i64, stride: i64) -> nn::Conv2D {
    let cfg = nn::ConvConfig {
        stride,
        padding,
        bias: false,
        ..Default::default()
    };
    nn::conv2d(p, c_in, c_out, 4, cfg)
}

fn generator(root: &nn::Path) -> impl nn::ModuleT {
    nn::seq_t()
        .add(tr2d(root / "tr1", LATENT_DIM, 1024, 0, 1))
        .add(nn::batch_norm2d(root / "bn1", 1024, Default::default()))
        .add_fn(Tensor::relu)
        .add(tr2d(root / "tr2", 1024, 512, 1, 2))
        .add(nn::batch_norm2d(root / "bn2", 512, Default::default()))
        .add_fn(Tensor::relu)
        .add(tr2d(root / "tr3", 512, 256, 1, 2))
        .add(nn::batch_norm2d(root / "bn3", 256, Default::default()))
        .add_fn(Tensor::relu)
        .add(tr2d(root / "tr4", 256, 128, 1, 2))
        .add(nn::batch_norm2d(root / "bn4", 128, Default::default()))
        .add_fn(Tensor::relu)
        .add(tr2d(root / "tr5", 128, 3, 1, 2))
        .add_fn(Tensor::tanh)
}

fn leaky_relu(v: f64) -> impl Fn(&Tensor) -> Tensor + Send + 'static {
    move |x| x.maximum(&(x * v))
}

fn discriminator(root: &nn::Path) -> impl nn::ModuleT {
    nn::seq_t()
        .add(conv2d(root / "conv1", 3, 128, 1, 2))
        .add_fn(leaky_relu(0.2))
        .add(conv2d(root / "conv2", 128, 256, 1, 2))
        .add(nn::batch_norm2d(root / "bn2", 256, Default::default()))
        .add_fn(leaky_relu(0.2))
        .add(conv2d(root / "conv3", 256, 512, 1, 2))
        .add(nn::batch_norm2d(root / "bn3", 512, Default::default()))
        .add_fn(leaky_relu(0.2))
        .add(conv2d(root / "conv4", 512, 1024, 1, 2))
        .add(nn::batch_norm2d(root / "bn4", 1024, Default::default()))
        .add_fn(leaky_relu(0.2))
        .add(conv2d(root / "conv5", 1024, 1, 0, 1))
}

fn mse_loss(x: &Tensor, y: &Tensor) -> Tensor {
    let diff = x - y;
    (&diff * &diff).mean(Kind::Float)
}

// Generate a 2D matrix of images from a tensor with multiple images.
fn image_matrix(imgs: &Tensor, sz: i64) -> Tensor {
    let imgs = ((imgs + 1.) * 127.5).clamp(0., 255.).to_kind(Kind::Uint8);
    let mut ys = Vec::new();
    for i in 0..sz {
        ys.push(Tensor::cat(
            &(0..sz)
                .map(|j| imgs.narrow(0, 4 * i + j, 1))
                .collect::<Vec<_>>(),
            2,
        ));
    }
    Tensor::cat(&ys, 3).squeeze_dim(0)
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::cuda_if_available();
    let args = std::env::args().collect::<Vec<_>>();
    let image_dir = match args.as_slice() {
        [_, d] => d.to_owned(),
        _ => panic!("usage: main image-dataset-dir"),
    };
    let images = tch::vision::image::load_dir(image_dir, IMG_SIZE, IMG_SIZE)?;
    println!("loaded dataset: {:?}", images);
    let train_size = images.size()[0];

    let random_batch_images = || {
        let index = Tensor::randint(train_size, &[BATCH_SIZE], (Kind::Int64, device));
        images.to_device(device)
            .index_select(0, &index)
            .to_kind(Kind::Float)
            / 127.5
            - 1.
    };
    let rand_latent =
        || Tensor::rand(&[BATCH_SIZE, LATENT_DIM, 1, 1], (Kind::Float, device)) * 2.0 - 1.0;

    let mut generator_vs = nn::VarStore::new(device);
    let generator = generator(&generator_vs.root());
    let mut opt_g = nn::adam(0.5, 0.999, 0.).build(&generator_vs, LEARNING_RATE)?;

    let mut discriminator_vs = nn::VarStore::new(device);
    let discriminator = discriminator(&discriminator_vs.root());
    let mut opt_d = nn::adam(0.5, 0.999, 0.).build(&discriminator_vs, LEARNING_RATE)?;

    let fixed_noise = rand_latent();

    for index in 0..BATCHES {
        discriminator_vs.unfreeze();
        generator_vs.freeze();
        let discriminator_loss = {
            let batch_images = random_batch_images();
            let y_pred = batch_images.apply_t(&discriminator, true);
            let y_pred_fake = rand_latent()
                .apply_t(&generator, true)
                .copy()
                .detach()
                .apply_t(&discriminator, true);
            mse_loss(&y_pred, &(y_pred_fake.mean(Kind::Float) + 1))
                + mse_loss(&y_pred_fake, &(y_pred.mean(Kind::Float) - 1))
        };
        opt_d.backward_step(&discriminator_loss);

        discriminator_vs.freeze();
        generator_vs.unfreeze();

        let generator_loss = {
            let batch_images = random_batch_images();
            let y_pred = batch_images.apply_t(&discriminator, true);
            let y_pred_fake = rand_latent()
                .apply_t(&generator, true)
                .apply_t(&discriminator, true);
            mse_loss(&y_pred, &(y_pred_fake.mean(Kind::Float) - 1))
                + mse_loss(&y_pred_fake, &(y_pred.mean(Kind::Float) + 1))
        };
        opt_g.backward_step(&generator_loss);

        if index % 1000 == 0 {
            let imgs = fixed_noise
                .apply_t(&generator, true)
                .view([-1, 3, IMG_SIZE, IMG_SIZE])
                .to_device(Device::Cpu);
            tch::vision::image::save(&image_matrix(&imgs, 4), format!("relout{}.png", index))?
        }
        if index % 100 == 0 {
            println!("{}", index)
        };
    }
    Ok(())
}
