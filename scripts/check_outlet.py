"""Optional tiny CPU check of the retained GRU/Transformer; no fitting/data IO."""
import json


def main():
    import numpy as np
    import mlx.core as mx
    from hydra_correctors.outlet.models_mlx import CorrectedModelConfig, build_model

    mx.set_default_device(mx.cpu)
    model = build_model("gru_transformer", "V3", 17,
                        CorrectedModelConfig(attention_backend="dense"))
    model.eval()
    # One synthetic example; no observation is marked available.
    history = np.zeros((1, 168, 10), dtype=np.float32)
    forecast = np.ones((1, 18), dtype=np.float32)
    errors = np.zeros((1, 18, 7), dtype=np.float32)
    site = np.array([0], dtype=np.int32)
    prediction = np.asarray(model(history, forecast, errors, site))
    replay = np.asarray(model(history, forecast, errors, site))
    assert prediction.shape == (1, 18) and np.isfinite(prediction).all()
    assert np.array_equal(prediction, replay), "Evaluation replay changed"
    try:
        model(history.astype(np.float64), forecast, errors, site)
    except ValueError:
        pass
    else:
        raise AssertionError("float64 input was silently accepted")
    print(json.dumps({"architecture": "gru_transformer", "variant": "V3",
                      "shape": list(prediction.shape), "parameters": model.parameter_count,
                      "device": "cpu", "same_input_replay": True,
                      "invalid_dtype_rejected": True, "forward_calls": 2,
                      "fits": 0, "data": "synthetic", "scientific_skill_claim": False}, indent=2))


if __name__ == "__main__":
    main()
