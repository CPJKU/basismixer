import numpy as np
import torch
import json
import basismixer.predictive_models


def test_full_model(model, seed=0):
    np.random.seed(seed)

    input_size = len(model.input_names)
    output_size = len(model.output_names)

    batch_size = 2
    seq_len = 5
    x = np.random.rand(seq_len, input_size).astype(np.float32)

    onsets = np.random.randint(0, seq_len, seq_len, dtype=np.int)
    onsets.sort()

    preds = model.predict(x, onsets)

    return preds


def main_test_save_load():
    m1_insize = 10
    m1_outsize = 3
    model1_config = dict(
        constructor=["basismixer.predictive_models", "FeedForwardModel"],
        args=dict(
            input_size=m1_insize,
            output_size=m1_outsize,
            hidden_size=5,
            input_names=["i{0}".format(i) for i in range(m1_insize)],
            output_names=["o{0}".format(i) for i in range(m1_outsize)],
        ),
    )

    m2_insize = 7
    m2_outsize = 2
    model2_config = dict(
        constructor=["basismixer.predictive_models", "RecurrentModel"],
        args=dict(
            input_size=m2_insize,
            output_size=m2_outsize,
            recurrent_size=7,
            hidden_size=5,
            n_layers=1,
            input_names=["i{0}".format(i + m1_insize) for i in range(m2_insize)],
            output_names=["o{0}".format(i + m1_outsize) for i in range(m2_outsize)],
            input_type="onsetwise",
        ),
    )

    model_configs = [model1_config, model2_config]
    input_names = [
        name for cfg in model_configs for name in cfg["args"]["input_names"]
    ] + ["not_in_input"]
    output_names = [
        name for cfg in model_configs for name in cfg["args"]["output_names"]
    ] + ["not_in_output"]
    full_model_config = dict(
        constructor=["basismixer.predictive_models", "FullPredictiveModel"],
        args=dict(
            models=model_configs,
            input_names=input_names,
            output_names=output_names,
            default_values=dict([(pn, 0) for pn in output_names]),
        ),
    )

    full_model = basismixer.predictive_models.construct_model(full_model_config)

    # make prediction
    preds_before = test_full_model(full_model)
    print("before")
    print(preds_before)

    # save config and params
    config_fn = "/tmp/fm_cfg.json"
    params_fn = "/tmp/fm_params.pkl"
    json.dump(full_model_config, open(config_fn, "w"), indent=2)
    torch.save(full_model.state_dict(), params_fn)

    # load config and params
    full_model_config = json.load(open(config_fn))
    full_model_params = torch.load(params_fn)
    full_model = basismixer.predictive_models.construct_model(
        full_model_config, full_model_params
    )

    # make prediction
    preds_after = test_full_model(full_model)
    print("after")
    print(preds_after)

    print(
        "predictions all close?",
        np.allclose(
            preds_before.view(np.float32),
            preds_after.view(np.float32),
        ),
    )


if __name__ == "__main__":
    main_test_save_load()
