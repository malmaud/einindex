# Einindex

Write indexing operations in an elegant syntax inspired by einstein notation.

## Usage
Simply import the `index` function:

```python
from einindex import index
```

## Examples

Given an array `x` with dimensions `i` and a set of indices `idx` with dimension `j`, `x[idx]` can be written as

```python
index("i, [i]j->i", x, idx)
```
---

Say we have an array `x` with dimensions `(i, j)` and indices `idx` with dimension `j` and are trying to compute the following `y`:

```python
y=torch.empty(i)
for i_index in range(i):
    y[i_index] = x[i_index, idx[i_index]]
```


This becomes
```python
index("i j, [j]i->i", x, idx)
```



