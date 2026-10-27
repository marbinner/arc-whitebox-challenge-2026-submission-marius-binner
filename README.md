# ARC White-Box Estimation Challenge 2026 — Phase 2 solution (team marius_binner)

This repository contains the source code and data files of our two final Phase 2 submissions to the
[ARC White-Box Estimation Challenge](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026).
The task is to predict the per-neuron mean post-ReLU activations of random ReLU MLPs (width 1024, depth 16,
He initialisation, Gaussian inputs) under a FLOP budget. The score is
final-layer MSE × max(0.1, FLOPs / 2^41); lower is better.

| Folder | Submission | Build | Public score | Folder sha256 |
|---|---|---|---|---|
| [`submission-337501/`](submission-337501/) | [#337501](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/337501) (pick 1) | CMP-1SC | 1.29915e-9 | `00bc59dfb2541a756e51ccef57696c1ad0603032018a0913280fe6ea8cc15eda` |
| [`submission-337556/`](submission-337556/) | [#337556](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/337556) (pick 2) | F0Z-1SC | 1.29525e-9 | `19cd0fcf68589de57edf99ec44021ef3193ed9f564c6c7d5d78c4ac1468ea4b2` |

Each folder is the exact submitted folder, byte for byte: `estimator.py` (the whole estimator in one file),
the precomputed data files it loads (`*.npz`: learned correction heads and lookup tables), and the
third-party credit and licence files. The two folders differ only in `estimator.py`.

## Requirements

- Python 3.10
- `numpy==2.2.6`, `flopscope==0.12.3`, `whestbench==0.16.1` (see `requirements.txt`). The grader ran the estimator
  as a flopscope 0.12.3 client against a flopscope 0.12.3 server; the full `flopscope` package also runs it in-process.

## Running it

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# score on official public networks (labels from 1e9 samples), streamed from Hugging Face
whest run --estimator submission-337501/estimator.py --no-residual-wall-time-limit \
    --dataset hf://aicrowd/arc-whestbench-public-2026 --revision v2-phase2 --split mini --streaming --n-mlps 4

# quick smoke test on freshly generated networks
whest run --estimator submission-337501/estimator.py --no-residual-wall-time-limit --n-mlps 1
```

On the first public network the first command gives an adjusted score of 1.50e-9 (raw final-layer MSE 1.25e-8, compute
utilisation 0.120); the official 100-network score of this submission is 1.29915e-9. The smoke test labels its networks
with only 200,000 Monte-Carlo samples, so its MSE (~3e-7) measures that sampling noise, not the estimator.

The estimator reads its data files from its own folder, so keep each folder intact. The submitted estimator class is
`Estimator`. On a single CPU core one network takes about 1.5–3 minutes in-process.

Why `--no-residual-wall-time-limit`: the grader allows 0.4 s of unmetered Python time per network and measured about
0.32 s for these submissions on its hardware (client/server mode). A local in-process run on a slower or busy machine
usually exceeds 0.4 s and would mark every network as failed, although the predictions and FLOP counts are the same.
On a slow machine you may also want `--wall-time-limit 600`.

## Switch-off ablations

`ablation/ablate.py` reproduces the switch-off table of the technical write-up (Table 4). It runs the exact submitted
estimator of #337501 with one component switched off, without retraining, on networks of the public `full` split
(labels from 1e9 samples). `ablation/summarize.py` then compares each variant with the unmodified estimator (`base`)
network by network: geometric-mean ratio of the noise-corrected final-layer MSE with a 95% confidence interval, the
number of networks that got worse, and the change in billed FLOPs.

```bash
mkdir -p results
# the 64 networks of the write-up are 896-959; `base` is the unmodified estimator
python ablation/ablate.py base   896 960 results/base.jsonl
python ablation/ablate.py r8_off 896 960 results/r8_off.jsonl
python ablation/summarize.py results/*.jsonl
```

`python ablation/ablate.py --help` lists the variants; Table 5 of the write-up gives the parameters each one switches
off. The first run downloads the prepared `full` split from Hugging Face (about 63 GB); `--dataset DIR` reads a local
copy of it instead. Runs resume from an existing output file, and a range of networks can be split over several
processes, each writing its own file.

## How it works (short)

The estimator propagates cumulants of the activations layer by layer instead of sampling: means, the full
covariance, a factored representation of third cumulants ("sources × legs", with low-rank compression of old
layers and "births" of skew created by intermediate ReLUs), a scale-mixture model of fourth cumulants, and an
Edgeworth readout. Three small learned heads correct the propagation inside the network; they were trained
end-to-end through a differentiable twin of the estimator. The arithmetic is kept inside the FLOP budget with
exact fast matrix multiplication schemes and exact restructuring. The full technical write-up will be linked here.

## Licence and credits

Our own code is released under the MIT licence (`LICENSE`). Each submission folder contains `NOTICE.txt`, which
credits every third-party component used, with the corresponding licence files:

- PLinOpt circuits (Dumas, Grenet, Pernet, Sedoglavic): CeCILL-B; the attribution notices are retained as that
  licence requires.
- Alternative-basis rank-7 circuits (Schwartz, Toledo, Vaknin, Wiernik): CC BY 4.0 article.
- FastMatrixMultiplication schemes (Andrew Perminov): MIT.
- AlphaEvolve <3,4,6> rank-54 scheme (Novikov et al., Google DeepMind): Apache-2.0 / CC BY 4.0.
- flip-cpd structured circuits (Khoruzhii, Gelß, Pokutta): MIT.
- ARC's cumulant-propagation reference code (alignment-research-center/mlp_cumulant_propagation): MIT.

## Use of AI tools

Most of this estimator's code, and the tooling and tests behind it, were written by AI coding agents working under
the team's direction and review: OpenAI Codex and ChatGPT, Anthropic Claude Code, and Google Gemini. The team chose
the research directions, set the acceptance tests and checked the results.
