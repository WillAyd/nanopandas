install nanobind

cmake -S . -B build
cmake --build build
cd build

python
>>> import nanopandas as npd
>>> npd.add(1, 2)