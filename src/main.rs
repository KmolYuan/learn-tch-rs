use clap::Parser;
use std::{error::Error, path::PathBuf};

mod eval;
mod model;
mod train;

#[derive(Parser)]
#[clap(
    name = "learn-tch-rs",
    version = env!("CARGO_PKG_VERSION"),
    author = env!("CARGO_PKG_AUTHORS"),
    about = env!("CARGO_PKG_DESCRIPTION"),
)]
struct Entry {
    #[clap(subcommand)]
    subcommand: Subcommand,
}

#[derive(clap::Subcommand)]
enum Subcommand {
    /// Start training process
    Train {
        /// Dataset path
        dataset: PathBuf,
        /// Model path
        model: PathBuf,
        /// Demo path
        demo: PathBuf,
        /// Total epoch
        #[clap(long, default_value = "10000")]
        epoch: u64,
    },
    /// Evaluate mode
    Eval {
        /// Generator path
        gen_path: PathBuf,
        /// Demo path
        demo: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Entry::parse();
    match args.subcommand {
        Subcommand::Train {
            dataset,
            model,
            demo,
            epoch,
        } => train::train(dataset, model, demo, epoch),
        Subcommand::Eval { gen_path, demo } => eval::eval(gen_path, demo),
    }
}
