#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>

#include "../array_types.hpp"

namespace nb = nanobind;

template <typename T> T FromSequence(nb::sequence sequence);

template <typename T> T FromFactorized(const Int64Array &locs, const T &values);

template <typename T>
std::optional<typename T::ScalarT> __getitem__(const T &self, int64_t i);

template <typename T> BoolArray __eq__(const T &self, const T &other);

template <typename T> std::string __repr__(const T &self);

template <typename T> int64_t __len__(const T &self);

template <typename T> const char *dtype([[maybe_unused]] const T &self);

template <typename T> int64_t nbytes(const T &self);

template <typename T> int64_t size(const T &self);

template <typename T> bool any(const T &self);

template <typename T> bool all(const T &self);

template <typename T> BoolArray isna(const T &self);

template <typename T>
T take(const T &self, const std::vector<int64_t> &indices);

template <typename T> T copy(const T &self);

template <typename T> T fillna(const T &self, typename T::ScalarT replacement);

template <typename T> T dropna(const T &self);

template <typename T>
std::vector<std::optional<typename T::ScalarT>> to_pylist(const T &self);
