# Learn tch-rs

PyTorch at Rust programming language.

## Environment

Requirements:

+ cuda (depends on torch)
+ libtorch: <https://pytorch.org/get-started/locally/> (C++ / Java version)

Extract libtorch zip file to `/usr/local/libtorch` or somewhere else.

Update environment variables in `.bashrc` or equivalent. Check cuda via using `nvcc` command.

```bash
export PATH="/usr/local/cuda/bin:$PATH"
export LIBTORCH="/usr/local/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib":$LD_LIBRARY_PATH
```

For Windows users, we need to update `PATH` and `LIBTORCH` environment variables from the Control Panel. As shown in the following commands, the location of each variable can be temporarily specified.

```powershell
$Env:LIBTORCH = "C:\libtorch"
$Env:Path += ";C:\libtorch\lib"
```

## Command Line Interface

Dataset: <https://github.com/AlexiaJM/RelativisticGAN> (`/images`)

Training:

```bash
cargo run --release -- train images/ model/ demo/
```

Testing:

```bash
cargo run --release -- eval model/gen_8000.pt demo/
```
