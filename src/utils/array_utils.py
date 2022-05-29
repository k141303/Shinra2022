from collections import defaultdict

import torch


def padding(array, pad, seq_len):
    if len(array) >= seq_len:
        return array
    return array + [pad] * (seq_len - len(array))


def decompose_array_tensors(arrays):
    decomposed_arrays = defaultdict(list)
    for array in arrays:
        for key, value in array.items():
            decomposed_arrays[key].append(value)

    for key, value in decomposed_arrays.items():
        if type(value[0]) is list:
            decomposed_arrays[key] = flatten(decomposed_arrays[key])
            continue
        decomposed_arrays[key] = torch.cat(decomposed_arrays[key], dim=0)

    return decomposed_arrays


def flatten(arrays):
    flat_array = []
    for array in arrays:
        flat_array += array
    return flat_array


def slide(array, window=510, dup=32):
    if len(array) <= window:
        return [array]
    return [array[i : i + window] for i in range(0, len(array) - dup, window - dup)]
