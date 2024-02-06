For usage you can simply ``pip install .``


If developing install nanobind then:

```sh
cmake -S . -B build
cmake --build build
cd build
```

You can then run the test suite from the build folder with ``python -m pytest ../tests``

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

Note that we use utf8proc for string handling:

```python
>>> import nanopandas as nanopd
>>> arr = nanopd.StringArray(["üàéµ"])
>>> arr.upper().to_pylist()
['ÜÀÉΜ']
>>> arr.capitalize().to_pylist()
['Üàéµ']

Developing with sanitizers can work. Try this cmake config from the project root:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_SANITIZERS=ON
cmake --build build
cd build
ASAN_OPTIONS="detect_leaks=0" LD_PRELOAD="$(gcc -print-file-name=libasan.so)" python -m pytest -s ../tests/
```
