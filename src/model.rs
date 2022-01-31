use tch::{nn, Device, Kind, Tensor};

const LATENT_DIM: i64 = 128;
pub const IMG_SIZE: i64 = 64;

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

pub fn generator(root: &nn::Path) -> impl nn::ModuleT {
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

pub fn discriminator(root: &nn::Path) -> impl nn::ModuleT {
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

pub fn mse_loss(x: &Tensor, y: &Tensor) -> Tensor {
    let diff = x - y;
    (&diff * &diff).mean(Kind::Float)
}

pub fn rand_latent(batch_size: i64, device: Device) -> Tensor {
    Tensor::rand(&[batch_size, LATENT_DIM, 1, 1], (Kind::Float, device)) * 2.0 - 1.0
}

// Generate a 2D matrix of images from a tensor with multiple images.
pub fn image_matrix(imgs: &Tensor, sz: i64) -> Tensor {
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
