import datetime
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from create_model import create_model

checkpoints_dir = Path(__file__).resolve().parent / "checkpoints"


def save_checkpoint(model, epoch, train_loss, val_loss, is_best_val_loss):
    if checkpoints_dir.exists():
        assert checkpoints_dir.is_dir()
    else:
        checkpoints_dir.mkdir()

    now = datetime.datetime.now(datetime.UTC)
    checkpoint_dir = checkpoints_dir / f"{now:%Y%m%dZ%H%M%S}"
    checkpoint_dir.mkdir()

    tensors_file = checkpoint_dir / "model.safetensors"
    save_file(model.state_dict(), tensors_file)

    meta = dict(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
    )
    meta_file = checkpoint_dir / "meta.json"
    meta_file.write_text(json.dumps(meta))

    symlink_target = Path(".") / checkpoint_dir.name
    if is_best_val_loss:
        best_path = checkpoints_dir / "best"
        best_path.unlink(missing_ok=True)
        best_path.symlink_to(symlink_target, target_is_directory=True)


def load_model(checkpoint_name):
    model, tokenizer = create_model()
    checkpoint_tensors = checkpoints_dir / checkpoint_name / "model.safetensors"
    state = load_file(checkpoint_tensors)
    model.load_state_dict(state)

    return model, tokenizer
