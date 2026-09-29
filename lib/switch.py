"""Switch-point features derived from word-level LID tags.

    Tokens:  Movie  bahut  acchi  thi  but  ending  was  terrible
    LID:       EN     HI     HI    HI   EN     EN     EN     EN
    Switch:     0      1      0     0    1      0      0      0

switch_i = 1 if LID_i != LID_{i-1} else 0, with switch_0 = 0 by convention.

By default tokens tagged `o` (punctuation, mentions, emoji, numbers) are
transparent: they never count as a switch themselves and do not reset the
"previous language", so `hin o eng` yields exactly one switch (on `eng`).
Pass ignore_other=False for the naive definition where `o` is a language.
"""

from typing import List

OTHER = "o"


def switch_points(tags: List[str], ignore_other: bool = True) -> List[int]:
    switches = [0] * len(tags)
    prev = None
    for i, tag in enumerate(tags):
        tag = tag.lower()
        if ignore_other and tag == OTHER:
            continue
        if prev is not None and tag != prev:
            switches[i] = 1
        prev = tag
    return switches


def switch_distances(switches: List[int], max_bucket: int = 3) -> List[int]:
    """Distance (in words) to the nearest switch point, bucketed to 0..max_bucket.

    0 = this word is a switch, 1 = adjacent, ..., max_bucket = max_bucket or further
    (also used when the sentence has no switch at all).
    """
    n = len(switches)
    inf = n + max_bucket + 1
    dist = [inf] * n
    # left-to-right pass
    last = None
    for i, s in enumerate(switches):
        if s:
            last = i
        if last is not None:
            dist[i] = i - last
    # right-to-left pass
    last = None
    for i in range(n - 1, -1, -1):
        if switches[i]:
            last = i
        if last is not None:
            dist[i] = min(dist[i], last - i)
    return [min(d, max_bucket) for d in dist]


def switch_stats(tags: List[str], ignore_other: bool = True) -> dict:
    sw = switch_points(tags, ignore_other)
    return {"n_tokens": len(tags), "n_switches": sum(sw),
            "switch_rate": sum(sw) / max(1, len(tags))}


def align_to_subwords(word_feature: List[int], word_ids: List, pad_value: int,
                      mode: str = "all") -> List[int]:
    """Broadcast a per-word feature to sub-word tokens.

    word_ids: output of `tokenizer(...).word_ids()`; None for special tokens.
    mode="all"   -> every sub-word of a word carries the word's feature.
    mode="first" -> only the first sub-word carries it, the rest get pad_value.
    """
    out = []
    prev = None
    for wid in word_ids:
        if wid is None:
            out.append(pad_value)
        elif mode == "first" and wid == prev:
            out.append(pad_value)
        else:
            out.append(word_feature[wid])
        prev = wid
    return out


if __name__ == "__main__":
    toks = "Movie bahut acchi thi but ending was terrible".split()
    tags = ["eng", "hin", "hin", "hin", "eng", "eng", "eng", "eng"]
    sw = switch_points(tags)
    print("tokens :", toks)
    print("tags   :", tags)
    print("switch :", sw)
    print("dist   :", switch_distances(sw))
    print("with o :", switch_points(["hin", "o", "eng", "o", "o", "hin"]),
          "(naive:", switch_points(["hin", "o", "eng", "o", "o", "hin"], ignore_other=False), ")")
