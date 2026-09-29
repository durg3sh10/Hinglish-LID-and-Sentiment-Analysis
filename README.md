# Language Identification and Sentiment Analysis from Hindi English Code-Mixed Texts

This project is under active development for partial fullfillment of grade requirement for course code CS613-NLP at Indian Institute of Technology Gandhinagar.

### Enviornment setup

This project require specified python libraries. Make sure to use python enviornment for the experiments.

```sh
python3 -m venv venv # create environment (one time)
source venv/bin/activate # activate your enviornment (linux)
pip install -r requirements.txt # install all required packages (one time)
```

If you install any new packages, make sure to add those in `requirements.txt` by using

```sh
pip freeze > requirements.txt
```
> This project has been tested on CUDA12.4

### Download dataset

For all the experiment the datasets, models and hyperparameters remain same. Please use the data generation script to download and process the SentiMix dataset. Alongside, use `eval.py` to generate the result. Use below command to download and process the dataset.

```sh
python3 get_sentimix.py # one time 
```

This will download all the required files from the remote server and process it to `csv` format and store them in your current folder's `dataset/SentiMix/`. For evaluation use 

```sh
python3 eval.py
```

For experiment specific details, use branches `exp0`, `exp1`, ...

---

## Experiment 2 - Switch Point + Sentiment (branch `exp2`)

**Research question.** If the model is given the language *switch point* information, does the sentiment score improve?

```
Tokens:  Movie  bahut  acchi  thi  but  ending  was  terrible
LID:       EN     HI     HI    HI   EN     EN     EN     EN
Switch:     0      1      0     0    1      0      0      0
```

`switch_i = 1 if LID_i != LID_{i-1} else 0` (`switch_0 = 0`). Tokens tagged `o` (punctuation, mentions,
emoji) are transparent by default, i.e. `hin o eng` is a single switch; use `--naive_switch` to change that.

**Model.** Any HuggingFace encoder (default `xlm-roberta-base`) followed by *switch-biased attention pooling*

    a_i = softmax( W_h h_i + W_s e_i^sw + b ),     s = sum_i a_i h_i,     logits = W_c s

where `e_i^sw` is a small learned embedding (16-d) of the switch feature. Variants compared (`--pooling`):

| `--pooling` | switch feature `e_i^sw`                                    |
|-------------|-------------------------------------------------------------|
| `none`      | *baseline*: plain attention pooling, no switch information   |
| `binary`    | embedding of the binary switch flag {0, 1}                   |
| `distance`  | embedding of bucketed distance to nearest switch {0,1,2,3+}  |
| `cls`/`mean`| extra reference points (no attention pooling)                |

Switch points come either from the gold SentiMix LID tags (`--switch_source gold`) or from a small
token-level LID tagger (`lib/lid_tagger.py`, EN/HI/O) that is trained once and then **frozen**
(`--switch_source predicted`); the sentiment model never back-propagates into the tagger.

### Layout

```
lib/data.py        build/load train, dev, test CSVs (re-uses the shared get_sentimix.py, unchanged)
lib/switch.py      switch points, distance buckets, word -> sub-word alignment
lib/lid_tagger.py  token-level LID tagger (fine-tuned encoder), writes {split}_pred_tags.csv
lib/model.py       SwitchAwareSentimentModel (encoder + switch-biased attention pooling)
lib/train.py       training loop; writes results/exp2/<run>.csv + ground.csv for evaluate_all.py
main.py            CLI gateway (all hyper-parameters)
run.sh             full reproduction: tagger + 2 x 3 sentiment runs + shared evaluation
```

### Run

```sh
cp .env.example .env            # DATA_HOME=dataset, HG_DATACARD=RTT1/SentiMix
python3 lib/data.py             # download + build train/dev/test (test labels are joined from test_labels_hinglish.txt)
python3 lib/lid_tagger.py       # 1. LID tagger  -> checkpoints/lid_xlm-roberta-base/
python3 main.py --pooling none                        # 2a. baseline
python3 main.py --pooling binary                      # 2b. binary switch embedding (gold LID)
python3 main.py --pooling distance                    # 2c. distance-to-switch embedding (gold LID)
python3 main.py --pooling binary --switch_source predicted --pred_tags_dir checkpoints/lid_xlm-roberta-base
cd results/exp2 && python3 ../../evaluate_all.py --ground ground.csv --pred-glob "*.csv"   # 3. shared eval
```

or simply `bash run.sh [MODEL_NAME]`. Smoke test on CPU/GPU with
`python3 main.py --max_train_samples 400 --max_eval_samples 200 --epochs 1`.
Large decoder models: `--model_name mistralai/Mistral-7B-Instruct-v0.3 --lora --load_in_4bit --batch_size 8`.

Each run writes `results/exp2/<run>_metrics.json` with dev/test accuracy and macro P/R/F1
(same formula as `evaluate_all.py`) plus a breakdown on sentences **with** vs **without** a switch.

### Results (xlm-roberta-base, 3 epochs, batch 32, lr 2e-5 / head 1e-3, max_len 128, 2 x RTX A5000)

Frozen LID tagger (`lib/lid_tagger.py`, 2 epochs): dev token-acc 0.918 / macro-F1 0.929, test token-acc 0.912 / macro-F1 0.922.

Sentiment on the SentiMix test split (2 944 roman sentences), macro P/R/F1 from the shared `evaluate_all.py`.
Rows with 3 seeds report mean +- std over seeds {42, 1, 2}; the others are seed 42 only.

| pooling | switch source | scorer | seeds | dev F1 | test acc | test P | test R | test F1 |
|---|---|---|---|---|---|---|---|---|
| none (baseline) | - | linear | 3 | 0.642 | 0.711 | 0.716 | 0.715 | 0.715 +- 0.004 |
| binary | gold | linear | 3 | 0.638 | 0.711 | 0.717 | 0.714 | 0.715 +- 0.003 |
| distance | gold | linear | 3 | 0.640 | 0.708 | 0.714 | 0.710 | 0.712 +- 0.004 |
| binary | predicted | linear | 1 | 0.639 | 0.701 | 0.706 | 0.704 | 0.705 |
| distance | predicted | linear | 1 | 0.639 | 0.707 | 0.713 | 0.708 | 0.711 |
| none (baseline) | - | mlp | 1 | 0.632 | 0.708 | 0.711 | 0.713 | 0.711 |
| binary | gold | mlp | 1 | 0.640 | 0.710 | 0.714 | 0.715 | 0.714 |
| distance | gold | mlp | 1 | 0.642 | 0.716 | 0.720 | 0.719 | **0.719** |

Test macro-F1 by number of switch points per sentence (`lib/analyze.py`, linear scorer, gold LID, mean over 3 seeds):

| pooling | 1-2 switches (n=916) | 3-5 (n=1640) | 6-9 (n=381) |
|---|---|---|---|
| none | 0.734 | 0.708 | 0.699 |
| binary | 0.732 | 0.708 | 0.705 |
| distance | 0.731 | 0.703 | 0.703 |

**Take-aways (so far).**
1. With `xlm-roberta-base`, adding switch-point information to attention pooling does **not** change test macro-F1 beyond seed noise (+-0.4 F1): baseline 0.715, binary 0.715, distance 0.712.
2. The only hint of a gain is on switch-heavy sentences (6-9 switches: +0.6 F1 for binary), and the best single run is `distance + mlp` scorer (0.719), but both are single-seed / small-n and need more seeds before we claim anything.
3. Using switch points from the frozen tagger instead of gold LID costs ~0-1 F1, i.e. tagger errors (91% token accuracy) do not wipe out the signal, there simply is not much signal to lose.
4. Dev F1 (~0.64) is consistently ~7 points below test F1 (~0.71); the SentiMix dev split is harder than the test split.

Next steps: more seeds for the `mlp` scorer, fuse `e_i^sw` into `h_i` (not only into the attention score), try
the contrastive / PESTO-style positional variants from the experiment notes, and repeat with the group's
7B models (`--lora --load_in_4bit`).

Notes: the `emt` (emoticon) tag of SentiMix is folded into `o`; a handful of rows with a non-numeric
id emitted by the shared parser (a token literally spelled `meta`) and one exact duplicate id per
training split are dropped in `lib/data.py`. The test split (2 944 roman rows) is unaffected.

### Authors

- Parth Dangi - [parthgdangi](https://github.com/parthgdangi)
- Nishant Sharma - [rockbnishant](https://github.com/Rockbnishant)
- Durgesh Mishra - [durg3sh10](https://github.com/durg3sh10)
- Rahul Kumawat - [rahulkumawat835](https://github.com/rahulkumawat835)
- Lavish Jangid - [lavish-j](https://github.com/lavish-j)
- Harshiddhi Pathak - [horikita-99](https://github.com/horikita-99)
- Anuj Tiwari - [anujjtiwari](https://github.com/anujjtiwari) 
- Harsh Krishnadev Dubey [Hrshhh](https://github.com/Hrshhh)
- Suvasish Das - [suvasish114](https://github.com/suvasish114)


### Contributors

<a href="https://github.com/suvasish114/Hinglish-LID-and-Sentiment-Analysis/graphs/contributors"><img src="https://contrib.rocks/image?repo=suvasish114/Hinglish-LID-and-Sentiment-Analysis"/></a> 
