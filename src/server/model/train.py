import os
import random
import time
import typing

import wandb
import jax
import numpy as np
import optax
import transformers
from jax import numpy as jnp
from jax.experimental import maps
from mesh_transformer.checkpoint import read_ckpt_lowmem
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
from model.constants import MODEL_PARAMS

os.makedirs("outputs", exist_ok=True)

run = wandb.init(
    project="okbh", entity="homebrewnlp"
)  # Initialize wandb before model to log all prints, instantiate before tokenizer to avoid forking after tokenizer init

GENERATED_LENGTH = 256
TOP_P = 0.9
TEMPERATURE = 0.8

train_test_split = 0.9
per_replica_batch = MODEL_PARAMS["per_replica_batch"]
cores_per_replica = MODEL_PARAMS["cores_per_replica"]
seq = MODEL_PARAMS["seq"]

MODEL_PARAMS["sampler"] = nucleaus_sample

mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
devices = np.array(jax.devices()).reshape(mesh_shape)

maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ("dp", "mp")), ())

tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

total_batch = per_replica_batch * jax.device_count() // cores_per_replica


network = CausalTransformer(MODEL_PARAMS)
network.state = read_ckpt_lowmem(network.state, "step_383500/", devices.shape[1])

with open("dataset/dataset.txt", "r") as f:
    data = f.read()

USERS = [
    "Meadiao",
    "bemxio",
    "amozi",
    '"Deleted User"',
    "Eren (LEGIT)",
    "Text101",
    "fishern",
    "Raven",
    "mes",
    "brooklerz",
    "ClashLuke",
]

print_chunks = 32
tokens = tokenizer(data, verbose=False, return_tensors="np")["input_ids"][0]
tokens = tokens[: tokens.shape[0] - tokens.shape[0] % ((seq + 1) * total_batch)].astype(
    np.int32
)
eval_prompt = tokens[len(tokens) - MODEL_PARAMS["seq"] + GENERATED_LENGTH + 16 :]
tokens = jnp.array(tokens.reshape((-1, 1, total_batch, seq + 1)))
train_src = tokens[: int(tokens.shape[0] * train_test_split) // 8 * 8, :, :, :-1]
train_tgt = tokens[: int(tokens.shape[0] * train_test_split) // 8 * 8, :, :, 1:]
test_src = tokens[int(tokens.shape[0] * train_test_split) // 8 * 8 :, :, :, :-1]
test_tgt = tokens[int(tokens.shape[0] * train_test_split) // 8 * 8 :, :, :, 1:]
test_src = test_src[: test_src.shape[0] // 8 * 8, 0]
test_tgt = test_tgt[: test_src.shape[0] // 8 * 8, 0]

users = [f"<{usr}>:" for usr in USERS]
prompts = [
    np.concatenate(
        [
            eval_prompt,
            tokenizer(f"\n-----\n{usr}", verbose=False, return_tensors="np")[
                "input_ids"
            ][0],
        ]
    )
    for usr in USERS
]
prompt_len = [len(p) for p in prompts]
print(prompt_len)
prompts = (
    np.stack([np.pad(p, ((seq - p.shape[0]), 0)) for p in prompts])
    .astype(np.int32)
    .reshape(-1, 1, seq)
)
prompts = jnp.array(prompts)

indices = list(range(train_src.shape[0]))


epoch = 0
while True:
    start = time.time()

    random.shuffle(indices)
    for outer_idx in range(print_chunks):
        start = time.time()
        train_loss = jnp.zeros(())
        test_loss = jnp.zeros(())
        train_steps = 0
        test_steps = 0
        for idx in range(
            int(train_src.shape[0] / print_chunks * outer_idx),
            int(train_src.shape[0] / print_chunks * (outer_idx + 1)),
        ):
            inp = train_src[indices[idx]]
            loss, _, _, _, network.state = network.train_xmap(
                network.state, inp, train_tgt[indices[idx]]
            )
            train_loss += loss
            train_steps += 1
        for idx in range(
            int(test_tgt.shape[0] / print_chunks * outer_idx),
            int(test_tgt.shape[0] / print_chunks * (outer_idx + 1)),
        ):
            test_loss += network.eval_xmap(
                network.state,
                test_src[idx],
                test_tgt[idx],
                np.array([test_src[idx].shape[1]] * test_src[idx].shape[0]),
            )["all_loss"]
            test_steps += 1
        train_loss = train_loss.mean() * total_batch / train_steps
        test_loss = test_loss.mean() * total_batch / test_steps

        step_time = time.time() - start
        print(
            f"Epoch: {epoch:3d} | Progress: {outer_idx:{len(str(print_chunks))}d}/{print_chunks} | TrainLoss {train_loss:7.4f} - TestLoss: {test_loss:7.4f} | "
            f"StepTime: {step_time:.1f}s"
        )
        run.log(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "step_time": step_time,
                "epoch": epoch,
                "local_step": outer_idx,
            }
        )

    ones = np.ones((1,))
    output = [
        network.generate(
            p,
            ones * pl,
            GENERATED_LENGTH,
            {"top_p": ones * TOP_P, "temp": ones * TEMPERATURE},
        )[0][0, :, 0]
        for p, pl in zip(prompts, prompt_len)
    ]
    with open(f"outputs/{epoch}.txt", "w") as f:
        f.write(
            "\n\n\n\n------------------------\n\n\n\n".join(
                usr + tokenizer.decode(o) for usr, o in zip(users, output)
            )
        )

    epoch += 1
    for shard_id in range(MODEL_PARAMS["cores_per_replica"]):
        network.write_ckpt(f"gs://ggpt4/kek/", shard_id)
