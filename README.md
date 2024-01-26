For usage you can simply ``pip install .``


If developing install nanobind then:

```sh
cmake -S . -B build
cmake --build build
cd build
```

Usage:

```python
>>> import nanopandas as nanopd
>>> arr = nanopd.StringArray(["foo", "bar", "baz", "baz", None])
>>> arr.size
5
>>> arr.nbytes
48
>>> arr.dtype
'large_string[nanoarrow]'
>>> arr.to_pylist()
['foo', 'bar', 'baz', 'baz', None]
>>> arr.unique().to_pylist()
['bar', 'baz', 'foo']
```