from train_style_disentangle import train_alternating
from config import get_opts
from data import set_seed
from pathlib import Path

if __name__ == "__main__":
    set_seed(1234)
    opts = get_opts()
    Path(opts.save_dir).mkdir(parents=True, exist_ok=True)
    train_alternating(opts)