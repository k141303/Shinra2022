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
        decomposed_arrays[key] = torch.cat(decomposed_arrays[key], dim=0)

    return decomposed_arrays
