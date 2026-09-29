Source: https://huggingface.co/Contrastive-LM/CLM-v0.1-8B
Retrieved: 2026-09-27

![CLM v0.1](https://raw.githubusercontent.com/Contrastive-LM/CLM/main/assets/logo.png)

### CLM-v0.1-8B

\| 📄 [**Blog**](https://contrastive-lm.notion.site/)
\| 💻 [**Code**](https://github.com/Contrastive-LM/CLM)
\| 🗣️ [**Discord**](https://discord.gg/5dAQEDJBs) \|

**Contrastive Language Model (CLM)** is a new class of **System One model**
trained with a **contrastive learning** objective that connects **states and**
**actions**. **CLM-8B** consists of two small projection heads (a state head and
an action head) on top of a frozen **Qwen3-8B** encoder trained with a
bidirectional InfoNCE loss.

- **Training:** pre-trained on ~60M Nemotron Q&A pairs, mid-trained on ~30M
synthetic hard negatives, post-trained on ~1M agentic trajectories.
- **Zero-shot:** on par with Jev on computer-use, gaming and tool-calling tasks,
with **up to 9× lower latency**.
- **Fine-tuned as a verifier:** SOTA on **DeepSWE (81.6%)** and
**Terminal-Bench 2.1 (87.6%)**, 4–6× faster than Jev.
- **State & Action Caching:** states and actions are encoded separately, so action
embeddings can be reused. **With ~1k candidates, CLM is 13× faster than Jev.**

## Usage

### With the `contrastive-lm` package

```bash
pip install contrastive-lm

# 1. encoder (Qwen3-8B embeddings)
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --max-model-len 2048 --port 8090 &

# 2. API + playground at http://localhost:8700/ (fetches CLM_v0.1-8B.pt into ~/.cache/clm/)
clm-serve
```

Ask typed questions about a state:

```python
from clm import CLMClient, Choice, Noul, Score

client = CLMClient()  # http://127.0.0.1:8700 by default
r = client.system_one(
    state="Customer: my invoice was charged twice and nobody answers the phone!",
    questions={
        "urgency": Noul(instructions="Is this urgent?"),
        "department": Choice(instructions="Which team should handle this?",
                             criteria={"billing": "Charges, invoices, refunds",
                                       "technical": "Bugs and outages"}),
        "frustration": Score(instructions="How frustrated is the customer?",
                             criteria=["Calm", "Frustrated", "Very angry"]),
    },
)
print(r.answers["department"].choice)         # billing
print(r.answers["department"].probabilities)  # {'billing': 0.93878, 'technical': 0.06122}
```

Or rank free-form candidates (best-of-N solutions, tool names, next moves):

```python
from clm import Engine

engine = Engine(emb_url="http://127.0.0.1:8090/v1/embeddings")
engine.rank("What causes tides on Earth?",
            ["The Moon's gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round."])
# [{'rank': 1, 'candidate': "The Moon's gravitational pull.", 'prob': 0.993}, ...]
```

### Fine-tuning

Only the heads are trained, so fine-tuning is cheap. This checkpoint is the
starting point for the DeepSWE and Terminal-Bench heads.

```bash
git clone https://github.com/Contrastive-LM/CLM.git && cd CLM && pip install -e .
hf download Contrastive-LM/deepswe-clm-heads-8k heldout_tasks.json --local-dir heads/deepswe
python train/finetune.py --task clm --init-ckpt "$(clm-download)" --out-dir runs/deepswe \
    --holdout-tasks heads/deepswe/heldout_tasks.json --batch 512
```

See the [fine-tuning guide](https://github.com/Contrastive-LM/CLM/blob/main/docs/FINETUNING.md).

### Playground

`clm-serve` also serves a web playground at `http://localhost:8700/`.

![The CLM playground: a state with three typed questions on the left, their answer distributions on the right](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/resolve/main/assets/playground.png)

## Limitations

- **Encoder-locked:** the heads require Qwen3-8B last-token-pooled embeddings.
- **No generation:** CLM only scores the candidates you give it, and its
probabilities are relative to that set.
- **Verifier results need fine-tuning:** the SOTA agentic-benchmark numbers
come from fine-tuned heads, not this checkpoint zero-shot.
- **Generalization:** CLM-8B is one rung of our scaling ladder. A multimodal
**CLM-35B**, trained with more data, compute and parameters for stronger
generalization, is coming in early October.

## Citation

```bibtex
@misc{kwok2026contrastivelanguagemodels,
  title={Contrastive Language Models: A System One Model for Fast and Generalizable Decision-Making},
  author={Jacky Kwok and Hangoo Kang and Tarun Suresh and Jon Saad-Falcon and Marco Pavone and Christopher Ré and Azalia Mirhoseini},
  year={2026},
  note={Notion Blog},
  url={https://contrastive-lm.notion.site}
}
```

## License

The CLM-8B weights are released under the [Apache 2.0 License](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/tree/main/LICENSE). The base encoder [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) is also Apache 2.0.

Downloads last month434

Inference Providers [NEW](https://huggingface.co/docs/inference-providers)

[Text Ranking](https://huggingface.co/tasks/text-ranking "Learn more about text-ranking")

This model isn't deployed by any Inference Provider. [🙋Ask for provider support](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/new?title=Contrastive-LM/CLM-v0.1-8B&description=React%20to%20this%20comment%20with%20an%20emoji%20to%20vote%20for%20%5BContrastive-LM%2FCLM-v0.1-8B%5D(%2FContrastive-LM%2FCLM-v0.1-8B)%20to%20be%20supported%20by%20Inference%20Providers.%0A%0A(optional)%20Which%20providers%20are%20you%20interested%20in%3F%20(Novita%2C%20Hyperbolic%2C%20Together%E2%80%A6)%0A)

## Model tree for Contrastive-LM/CLM-v0.1-8B

Base model

[Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base)

Finetuned

[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

Finetuned

( [2108](https://huggingface.co/models?other=base_model:finetune:Qwen/Qwen3-8B))

this model

Finetunes

[1 model](https://huggingface.co/models?other=base_model:finetune:Contrastive-LM/CLM-v0.1-8B)

## Spaces using Contrastive-LM/CLM-v0.1-8B2

Inference providers allow you to run inference using different serverless providers.

StripeM-Inner