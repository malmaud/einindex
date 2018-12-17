# Einindex

Write indexing operations in an elegant syntax inspired by einstein notation.

## Usage

Install with `pip install einindex`.

Then simply import the `index` function in a Python session:

```python
from einindex import index
```

## Examples

Given an array `x` with dimensions `i` and a set of indices `idx` with dimension `j`, `x[idx]` can be written as

```python
index(x, idx, "i, [i]j->j")
```
---

Say we have an array `x` with dimensions `(i, j)` and indices `idx` with dimension `j` and are trying to compute the following `y`:

```python
y = torch.empty(i)
for i_index in range(i):
    y[i_index] = x[i_index, idx[i_index]]
```


This becomes
```python
index(x, idx, "i j, [j]i->i")
```


---
Limited support for multindexing is present.

Imagine if `x` is a batch of images with dimensions `(batch, width, height)` and we are trying to pick out one pixel from each image, so the indexing array has dimensions `(batch, 2)` and we want to compute `y` where

```python
y = torch.empty(batch)
for batch_idx in range(batch):
    y[batch_idx] = x[batch_idx, idx[batch_idx, 0], idx[batch_idx, 1]]
```

This becomes

```python
index(x, idx, "batch width height, [width height] batch->batch")
```
