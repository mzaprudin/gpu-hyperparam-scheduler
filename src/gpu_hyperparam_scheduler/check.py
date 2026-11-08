"""Check dependencies and run an optional real CPU model smoke test."""

import argparse
import importlib
import json
import os
from pathlib import Path
import pkgutil
import tempfile


def smoke():
    import numpy as np
    import tensorflow as tf
    from .create_model import ModelParameters, compile_model, create_optimized_cnn_lstm_transformer, load_model
    from .data_generator import MasterGenerator, MultiPersonTimeSeriesGenerator
    from .demo import make_data

    tf.keras.utils.set_random_seed(42)
    frame = make_data(series=1, rows=48)["synthetic-0"]
    frame["filter"] = True
    frame["y_class"] = np.arange(len(frame)) % 2
    master = MasterGenerator([frame], ["price", "buy_volume", "sell_volume"], window_size=16)
    generator = MultiPersonTimeSeriesGenerator(master, batch_size=8)
    x, y = generator[0]
    params = ModelParameters(num_heads=2, head_size=16, ff_dim=32,
                             num_transformer_blocks=1, mlp_units=[16, 8])
    with tf.device("/CPU:0"):
        model = compile_model(create_optimized_cnn_lstm_transformer(16, 3, params), params)
        loss = model.train_on_batch(x, y)
        prediction = model(x, training=False).numpy()
        if not np.isfinite(loss).all() or not np.isfinite(prediction).all():
            raise RuntimeError("Non-finite loss or prediction")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "smoke.keras"
            model.save(checkpoint)
            restored = load_model(checkpoint)
            np.testing.assert_allclose(restored(x, training=False).numpy(), prediction, rtol=1e-5, atol=1e-6)
    return {"samples": len(x), "parameters": model.count_params(), "loss": float(loss[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run CPU forward/backward pass and checkpoint round trip")
    args = parser.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if args.smoke:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    import gpu_hyperparam_scheduler

    imported = []
    for module in pkgutil.walk_packages(gpu_hyperparam_scheduler.__path__, gpu_hyperparam_scheduler.__name__ + "."):
        importlib.import_module(module.name)
        imported.append(module.name)
    report = {"tensorflow": tf.__version__, "modules_imported": len(imported),
              "gpus": [device.name for device in tf.config.list_physical_devices("GPU")]}
    if args.smoke:
        report["cpu_smoke"] = smoke()
    print(json.dumps(report, indent=2))
    return 0
