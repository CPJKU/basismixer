import numpy as np
import torch
import basismixer.predictive_models as models

if __name__ == "__main__":

    input_size1 = 10
    output_size1 = 3
    recurrent_size = 7
    hidden_size = 5
    n_layers = 1

    input_names1 = ["i{0}".format(i) for i in range(input_size1)]
    output_names1 = ["o{0}".format(i) for i in range(output_size1)]
    model1 = models.FeedForwardModel(
        input_size=input_size1,
        output_size=output_size1,
        hidden_size=hidden_size,
        input_names=input_names1,
        output_names=output_names1,
    )

    model1.type(torch.float32)

    input_size2 = 7
    output_size2 = 2
    input_names2 = ["i{0}".format(i + input_size1) for i in range(input_size2)]
    output_names2 = ["o{0}".format(i + output_size1) for i in range(output_size2)]
    model2 = models.RecurrentModel(
        input_size=input_size2,
        output_size=output_size2,
        recurrent_size=recurrent_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        input_names=input_names2,
        output_names=output_names2,
        input_type="onsetwise",
    )
    # model2.type(torch.float32)

    input_names = input_names1 + input_names2 + ["not_in_input"]
    output_names = output_names1 + output_names2 + ["not_in_output"]
    model = models.FullPredictiveModel(
        [model1, model2],
        input_names,
        output_names,
        default_values=dict([(pn, 0) for pn in output_names]),
    )

    input_size = len(input_names)
    output_size = len(output_names)

    batch_size = 2
    seq_len = 5

    x = np.random.rand(seq_len, input_size).astype(np.float32)

    onsets = np.random.randint(0, seq_len, seq_len, dtype=np.int)
    onsets.sort()

    preds = model.predict(x, onsets)

    # x = torch.rand(batch_size, seq_len, input_size)

    # target = torch.rand(batch_size, seq_len, output_size)

    # mask = torch.ones(batch_size, seq_len)
    # y = model(x)

    # loss = models.recurrent_loss(y, target, mask)

    # preds = model.predict(x)
