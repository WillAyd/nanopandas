#pragma once

#include <cstring>
#include <functional>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <utf8proc.h>

#include "../array_types.hpp"

namespace nb = nanobind;

// similar to BitmapInvert from pandas PR #54506 but less safe
static int InvertInplace(uint8_t *buf, size_t nbytes) {
  size_t int64_strides = nbytes / sizeof(int64_t);
  size_t rem = nbytes % sizeof(int64_t);

  uint8_t *cursor = buf;
  for (size_t i = 0; i < int64_strides; i++) {
    int64_t value;
    memcpy(&value, cursor, sizeof(int64_t));
    value = ~value;
    memcpy(cursor, &value, sizeof(int64_t));
    cursor += sizeof(int64_t);
  }

  for (size_t i = 0; i < rem; i++) {
    uint8_t value;
    memcpy(&value, cursor, sizeof(uint8_t));
    value = ~value;
    memcpy(cursor, &value, sizeof(uint8_t));
    cursor++;
  }

  return 0;
}

template <typename T>
T FromSequence([[maybe_unused]] const T &self, nb::sequence sequence) {
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), T::ArrowT)) {
    throw std::runtime_error("Unable to init output array for FromSequence!");
  }

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  for (const auto &item : sequence) {
    if (item.is_none()) {
      if (ArrowArrayAppendNull(result.get(), 1)) {
        throw std::runtime_error("failed to append null!");
      }
    } else {
      if constexpr (std::is_same_v<T, BoolArray> ||
                    std::is_same_v<T, Int64Array>) {
        auto value = nb::cast<typename T::ScalarT>(item);
        if (ArrowArrayAppendInt(result.get(), value)) {
          throw std::runtime_error("failed to append int value!");
        }
      } else if constexpr (std::is_same_v<T, StringArray>) {
        std::string_view sv = nb::cast<std::string_view>(item);
        const struct ArrowStringView arrow_sv = {
            sv.data(), static_cast<int64_t>(sv.size())};
        if (ArrowArrayAppendString(result.get(), arrow_sv)) {
          throw std::runtime_error("failed to append string!");
        }
      } else {
        // see https://stackoverflow.com/a/64354296/621736
        static_assert(!sizeof(T), "FromSequence not implemented for type");
      }
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return T(std::move(result));
}

template <typename T>
T FromFactorized([[maybe_unused]] const T &self, const Int64Array &locs,
                 const T &values) {
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), T::ArrowT)) {
    throw std::runtime_error("Unable to init output array for FromFactorized!");
  }
  const auto n = locs.array_view_->length;

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not append!");
  }

  if (ArrowArrayReserve(result.get(), n)) {
    throw std::runtime_error("Unable to reserve array!");
  }

  for (int64_t idx = 0; idx < n; idx++) {
    const auto loc_value =
        ArrowArrayViewGetIntUnsafe(locs.array_view_.get(), idx);
    if (loc_value == -1) {
      if (ArrowArrayAppendNull(result.get(), 1)) {
        throw std::invalid_argument("Failed to append null!");
      }
    } else {
      const auto value = T::ArrowGetFunc(values.array_view_.get(), loc_value);
      if (T::ArrowAppendFunc(result.get(), value)) {
        throw std::runtime_error("Append call failed!");
      }
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return T(std::move(result));
}

template <typename T>
auto GetItemDunderInternal(const T &self, int64_t index)
    -> std::optional<typename T::ArrowScalarT> {
  const auto n = self.array_view_.get()->length;
  if ((index >= n) || (index < -n)) {
    throw std::out_of_range("index out of bounds");
  }

  const auto idx = index >= 0 ? index : n + index;
  if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
    return std::nullopt;
  }

  return T::ArrowGetFunc(self.array_view_.get(), idx);
}

template <typename T>
auto GetItemDunder(const T &self, nb::object indexer) -> nb::object {

  // Attempt 1. - scalar
  int64_t i;
  if (nb::try_cast(indexer, i, false)) {
    if (const auto result = GetItemDunderInternal(self, i)) {
      if constexpr (std::is_same_v<T, BoolArray> ||
                    std::is_same_v<T, Int64Array>) {
        return typename T::PyObjectT(*result);
      } else if constexpr (std::is_same_v<T, StringArray>) {
        return typename T::PyObjectT(result->data, result->size_bytes);
      } else {
        // see https://stackoverflow.com/a/64354296/621736
        static_assert(!sizeof(T), "__getitem__ not implemented for type");
      }
    } else {
      return nb::none();
    }
  }

  // At this point we are working with an iterable
  // TODO: we are falling back to return Python containers, but ideally we
  // should still return T wrapped as a Python object
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), T::ArrowT)) {
    throw std::runtime_error("Unable to init output array for take!");
  }

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  // Attempt 2. - list of ints
  std::vector<std::optional<int64_t>> values;
  if (nb::try_cast(indexer, values, false)) {
    for (const auto idx : values) {
      if (idx) {
        if (const auto value = GetItemDunderInternal(self, *idx)) {
          if (T::ArrowAppendFunc(result.get(), *value)) {
            throw std::runtime_error("Append call failed!");
          }
        } else {
          if (ArrowArrayAppendNull(result.get(), 1)) {
            throw std::runtime_error("failed to append null!");
          }
        }
      } else {
        if (ArrowArrayAppendNull(result.get(), 1)) {
          throw std::runtime_error("failed to append null!");
        }
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    nb::handle py_type = nb::type<T>();
    T *out = new T(std::move(result));
    return nb::inst_take_ownership(py_type, out);
  }

  // Attempt 3. - ndarray
  nb::ndarray<const bool, nb::ndim<1>> array;
  if (nb::try_cast(indexer, array, false)) {
    auto v = array.view();
    for (size_t idx = 0; idx < v.shape(0); idx++) {
      const auto should_index = v(idx);
      if (should_index) {
        if (const auto value = GetItemDunderInternal(self, idx)) {
          if (T::ArrowAppendFunc(result.get(), *value)) {
            throw std::runtime_error("Append call failed!");
          }
        } else {
          if (ArrowArrayAppendNull(result.get(), 1)) {
            throw std::runtime_error("failed to append null!");
          }
        }
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    nb::handle py_type = nb::type<T>();
    T *out = new T(std::move(result));
    return nb::inst_take_ownership(py_type, out);
  }

  // Attempt 4. - slice
  nb::slice sliceobj;
  if (nb::try_cast(indexer, sliceobj, false)) {
    const auto converted_slice = sliceobj.compute(self.array_view_->length);
    const auto [start, _, step, slice_length] = converted_slice;

    auto idx = start;
    for (size_t i = 0; i < slice_length; i++) {
      if (const auto value = GetItemDunderInternal(self, idx)) {
        if (T::ArrowAppendFunc(result.get(), *value)) {
          throw std::runtime_error("Append call failed!");
        }
      } else {
        if (ArrowArrayAppendNull(result.get(), 1)) {
          throw std::runtime_error("failed to append null!");
        }
      }
      idx += step;
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    nb::handle py_type = nb::type<T>();
    T *out = new T(std::move(result));
    return nb::inst_take_ownership(py_type, out);
  }

  throw std::out_of_range(
      "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
      "(`None`) and integer or boolean arrays are valid indices");
}

template <typename T> BoolArray EqDunder(const T &self, const T &other) {
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_BOOL)) {
    throw std::runtime_error("Unable to init bool array!");
  }
  const auto n = self.array_view_->length;

  if (n != other.array_view_->length) {
    throw std::range_error("Arrays are not of equal size");
  }

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), n)) {
    throw std::runtime_error("Unable to reserve array!");
  }

  for (int64_t i = 0; i < n; i++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), i) ||
        ArrowArrayViewIsNull(other.array_view_.get(), i)) {
      if (ArrowArrayAppendNull(result.get(), 1)) {
        throw std::runtime_error("failed to append null!");
      }
    } else {
      bool is_equal = false;
      if constexpr (std::is_same_v<T, BoolArray> ||
                    std::is_same_v<T, Int64Array>) {
        const auto left = ArrowArrayViewGetIntUnsafe(self.array_view_.get(), i);
        const auto right =
            ArrowArrayViewGetIntUnsafe(other.array_view_.get(), i);
        is_equal = left == right;
      } else if constexpr (std::is_same_v<T, StringArray>) {
        const auto left =
            ArrowArrayViewGetStringUnsafe(self.array_view_.get(), i);
        const auto right =
            ArrowArrayViewGetStringUnsafe(other.array_view_.get(), i);
        const auto nbytes = left.size_bytes;
        is_equal =
            (nbytes == right.size_bytes) &&
            (!strncmp(left.data, right.data, static_cast<size_t>(nbytes)));
      } else {
        // see https://stackoverflow.com/a/64354296/621736
        static_assert(!sizeof(T), "__eq__ not implemented for type");
      }

      if (is_equal) {
        if (ArrowArrayAppendInt(result.get(), 1)) {
          throw std::runtime_error("failed to append true value!");
        }
      } else {
        if (ArrowArrayAppendInt(result.get(), 0)) {
          throw std::runtime_error("failed to append false value!");
        }
      }
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return result;
}

template <typename T> std::string ReprDunder(const T &self) {
  std::ostringstream out{};
  out << T::Name << "\n[";

  const auto n = self.array_view_->length;
  for (int64_t idx = 0; idx < n; idx++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
      out << "null";
    } else {

      if constexpr (std::is_same_v<T, BoolArray>) {
        const auto value =
            ArrowArrayViewGetIntUnsafe(self.array_view_.get(), idx);
        if (value) {
          out << "True";
        } else {
          out << "False";
        }
      } else if constexpr (std::is_same_v<T, Int64Array>) {
        const auto value =
            ArrowArrayViewGetIntUnsafe(self.array_view_.get(), idx);
        out << std::to_string(value);
      } else if constexpr (std::is_same_v<T, StringArray>) {
        out << "\"";
        const auto arrow_sv =
            ArrowArrayViewGetStringUnsafe(self.array_view_.get(), idx);

        const auto nbytes = static_cast<size_t>(arrow_sv.size_bytes);
        const std::string_view sv{arrow_sv.data, nbytes};
        out << sv;
        out << "\"";
      } else {
        // see https://stackoverflow.com/a/64354296/621736
        static_assert(!sizeof(T), "__repr__ not implemented for type");
      }
    }
    if (idx < n - 1) {
      out << ", ";
    } else {
      out << "]";
    }
  }

  return out.str();
}

template <typename T> int64_t LenDunder(const T &self) {
  return self.array_view_->length;
}

template <typename T> ExtensionDtype<T> Dtype([[maybe_unused]] const T &self) {
  return ExtensionDtype<T>{};
}

template <typename T> int64_t Nbytes(const T &self) {
  const struct ArrowBufferView data_buffer =
      self.array_view_.get()->buffer_views[1];
  return data_buffer.size_bytes;
}

template <typename T> std::tuple<int64_t> Shape(const T &self) {
  return std::make_tuple(self.array_view_->length);
}

template <typename T> int64_t Size(const T &self) {
  return self.array_view_->length;
}

template <typename T> int64_t NullCount(const T &self) {
  return self.array_view_->null_count;
}

template <typename T> bool Any(const T &self) {
  return self.array_view_->length > self.array_view_->null_count;
}

template <typename T> bool All(const T &self) {
  return self.array_view_->null_count == 0;
}

template <typename T> BoolArray IsNA(const T &self) {
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_BOOL)) {
    throw std::runtime_error("Unable to init bool array!");
  }
  const auto n = self.array_view_->length;
  const int64_t bytes_required = _ArrowBytesForBits(n);

  const uint8_t *src = self.array_view_->buffer_views[0].data.as_uint8;
  struct ArrowBuffer *buffer = ArrowArrayBuffer(result.get(), 1);

  if (src == nullptr) {
    if (ArrowBufferAppendFill(buffer, 255, bytes_required)) {
      throw std::runtime_error("ArrowBufferAppendFill failed");
    }
  } else {
    if (ArrowBufferReserve(buffer, bytes_required)) {
      throw std::runtime_error("Could not reserve arrow buffer");
    }
    ArrowBufferAppendUnsafe(buffer, src, bytes_required);
  }

  result->length = n;
  result->null_count = 0;

  // Would be more efficient to iterate by word size. See BitmapInvert from
  // pandas PR #54506
  if (InvertInplace(buffer->data, bytes_required)) {
    throw std::runtime_error("Unexpected error with InvertInplace");
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return BoolArray(std::move(result));
}

template <typename T>
T Take(const T &self, const std::vector<int64_t> &indices) {
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), T::ArrowT)) {
    throw std::runtime_error("Unable to init output array for take!");
  }

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), indices.size())) {
    throw std::runtime_error("Unable to reserve array!");
  }

  const auto n = self.array_view_.get()->length;

  for (const auto index : indices) {
    if ((index >= n) || (index < -n)) {
      throw std::range_error("index out of bounds!");
    }

    const auto idx = index >= 0 ? index : n + index;
    if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
      if (ArrowArrayAppendNull(result.get(), 1)) {
        throw std::runtime_error("failed to append null!");
      }
    } else {
      const auto value = T::ArrowGetFunc(self.array_view_.get(), idx);
      if (T::ArrowAppendFunc(result.get(), value)) {
        throw std::runtime_error("Append call failed!");
      }
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return T(std::move(result));
}

template <typename T> T Copy(const T &self) {
  // This implementation is pretty naive; could be a lot faster if we
  // just memcpy the required buffers
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), T::ArrowT)) {
    throw std::runtime_error("Unable to init output for copy!");
  }
  const auto n = self.array_view_->length;

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), n)) {
    throw std::runtime_error("Unable to reserve array!");
  }

  for (int64_t idx = 0; idx < n; idx++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
      if (ArrowArrayAppendNull(result.get(), 1)) {
        throw std::runtime_error("failed to append null!");
      }
    } else {
      const auto value = T::ArrowGetFunc(self.array_view_.get(), idx);
      if (T::ArrowAppendFunc(result.get(), value)) {
        throw std::runtime_error("Append call failed!");
      }
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return T(std::move(result));
}

template <typename T> T FillNA(const T &self, typename T::ScalarT replacement) {
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), T::ArrowT)) {
    throw std::runtime_error("Unable to init output array for fillna!");
  }

  const auto n = self.array_view_->length;

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), n)) {
    throw std::runtime_error("Unable to reserve array!");
  }

  if constexpr (std::is_same_v<T, BoolArray> || std::is_same_v<T, Int64Array>) {
    for (int64_t idx = 0; idx < self.array_view_.get()->length; idx++) {
      if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
        if (ArrowArrayAppendInt(result.get(), replacement)) {
          throw std::runtime_error("failed to append int!");
        }
      } else {
        const auto sv = ArrowArrayViewGetIntUnsafe(self.array_view_.get(), idx);
        if (ArrowArrayAppendInt(result.get(), sv)) {
          throw std::runtime_error("failed to append int!");
        }
      }
    }
  } else if constexpr (std::is_same_v<T, StringArray>) {
    const struct ArrowStringView replacement_sv = {
        replacement.data(), static_cast<int64_t>(replacement.size())};
    for (int64_t idx = 0; idx < self.array_view_.get()->length; idx++) {
      if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
        if (ArrowArrayAppendString(result.get(), replacement_sv)) {
          throw std::runtime_error("failed to append string!");
        }
      } else {
        const auto sv =
            ArrowArrayViewGetStringUnsafe(self.array_view_.get(), idx);
        if (ArrowArrayAppendString(result.get(), sv)) {
          throw std::runtime_error("failed to append string!");
        }
      }
    }
  } else {
    // see https://stackoverflow.com/a/64354296/621736
    static_assert(!sizeof(T), "fillna not implemented for type");
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return T(std::move(result));
}

template <typename T> T DropNA(const T &self) {
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), T::ArrowT)) {
    throw std::runtime_error("Unable to init dropna output array!");
  }

  const auto n = self.array_view_->length - self.array_view_->null_count;
  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), n)) {
    throw std::runtime_error("Unable to reserve array!");
  }

  for (int64_t idx = 0; idx < self.array_view_.get()->length; idx++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
      continue;
    } else {
      const auto value = T::ArrowGetFunc(self.array_view_.get(), idx);
      if (T::ArrowAppendFunc(result.get(), value)) {
        throw std::runtime_error("Append call failed!");
      }
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return T(std::move(result));
}

template <typename T> T Interpolate(const T &self) {
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), T::ArrowT)) {
    throw std::runtime_error("Unable to init output for interpolate!");
  }

  // TODO: check for overflow
  const auto n = self.array_view_->length;

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), n)) {
    throw std::runtime_error("Unable to reserve array!");
  }

  bool seen_value = false;
  typename T::ArrowScalarT last_value_seen;

  for (int64_t idx = 0; idx < self.array_view_.get()->length; idx++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
      if (!seen_value) {
        if (ArrowArrayAppendNull(result.get(), 1)) {
          throw std::runtime_error("failed to append null!");
        }
      } else {
        if (T::ArrowAppendFunc(result.get(), last_value_seen)) {
          throw std::runtime_error("Append call failed!");
        }
      }
    } else {
      seen_value = true;
      const auto value = T::ArrowGetFunc(self.array_view_.get(), idx);
      if (T::ArrowAppendFunc(result.get(), value)) {
        throw std::runtime_error("Append call failed!");
      }
      last_value_seen = value;
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return T(std::move(result));
}

template <typename T> T PadOrBackfill(const T &self, std::string_view method) {
  if ((method != "pad") && (method != "backfill")) {
    throw std::invalid_argument("'method' must be either 'pad' or 'backfill'");
  }

  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), T::ArrowT)) {
    throw std::runtime_error("Unable to init large string array!");
  }

  const auto n = self.array_view_->length;

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), n)) {
    throw std::runtime_error("Unable to reserve array!");
  }

  if (method == "pad") {
    bool seen_value = false;
    typename T::ArrowScalarT last_value_seen;
    for (int64_t idx = 0; idx < self.array_view_.get()->length; idx++) {
      if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
        if (!seen_value) {
          if (ArrowArrayAppendNull(result.get(), 1)) {
            throw std::runtime_error("failed to append null!");
          }
        } else {
          if (T::ArrowAppendFunc(result.get(), last_value_seen)) {
            throw std::runtime_error("Append call failed!");
          }
        }
      } else {
        seen_value = true;
        const auto value = T::ArrowGetFunc(self.array_view_.get(), idx);
        if (T::ArrowAppendFunc(result.get(), value)) {
          throw std::runtime_error("Append call failed!");
        }
        last_value_seen = value;
      }
    }
  } else {

    int64_t last_append = 0;
    for (int64_t idx = 0; idx < self.array_view_.get()->length; idx++) {
      if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
        if (idx == self.array_view_.get()->length - 1) {
          do {
            if (ArrowArrayAppendNull(result.get(), 1)) {
              throw std::runtime_error("failed to append null!");
            }
            last_append++;
          } while (last_append <= idx);
        }
        continue;
      } else {
        const auto value = T::ArrowGetFunc(self.array_view_.get(), idx);
        do {
          if (T::ArrowAppendFunc(result.get(), value)) {
            throw std::runtime_error("Append call failed!");
          }
          last_append++;
        } while (last_append <= idx);
      }
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return T(std::move(result));
}

template <typename T> T Unique(const T &self) {
  std::set<typename T::ScalarT> uniques;
  const auto n = self.array_view_->length;

  for (int64_t i = 0; i < n; i++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), i)) {
      continue;
    }

    if constexpr (std::is_same_v<T, BoolArray> ||
                  std::is_same_v<T, Int64Array>) {
      const auto value = ArrowArrayViewGetIntUnsafe(self.array_view_.get(), i);
      uniques.insert(value);
    } else if constexpr (std::is_same_v<T, StringArray>) {
      const auto sv = ArrowArrayViewGetStringUnsafe(self.array_view_.get(), i);
      const std::string_view value{sv.data, static_cast<size_t>(sv.size_bytes)};
      uniques.insert(value);
    } else {
      // see https://stackoverflow.com/a/64354296/621736
      static_assert(!sizeof(T), "Unique not implemented for type");
    }
  }

  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), T::ArrowT)) {
    throw std::runtime_error("Unable to init large string array!");
  }

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), uniques.size())) {
    throw std::runtime_error("Unable to reserve array!");
  }

  // TODO: can make generic if we hash ArrowString
  for (const auto &val : uniques) {
    if constexpr (std::is_same_v<T, BoolArray> ||
                  std::is_same_v<T, Int64Array>) {
      if (ArrowArrayAppendInt(result.get(), val)) {
        throw std::runtime_error("failed to append int!");
      }
    } else if constexpr (std::is_same_v<T, StringArray>) {
      const struct ArrowStringView sv {
        val.data(), static_cast<int64_t>(val.size())
      };
      if (ArrowArrayAppendString(result.get(), sv)) {
        throw std::runtime_error("failed to append string!");
      }
    } else {
      // see https://stackoverflow.com/a/64354296/621736
      static_assert(!sizeof(T), "Unique not implemented for type");
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return T(std::move(result));
}

template <typename T> std::tuple<Int64Array, T> Factorize(const T &self) {
  std::unordered_map<typename T::ScalarT, int64_t> first_occurances;

  nanoarrow::UniqueArray values;
  if (ArrowArrayInitFromType(values.get(), T::ArrowT)) {
    throw std::runtime_error("Unable to init array for values!");
  }
  nanoarrow::UniqueArray locs;
  if (ArrowArrayInitFromType(locs.get(), NANOARROW_TYPE_INT64)) {
    throw std::runtime_error("Unable to init int64 array!");
  }
  const auto n = self.array_view_->length;

  if (ArrowArrayStartAppending(values.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayStartAppending(locs.get())) {
    throw std::runtime_error("Could not start appending");
  }

  for (int64_t idx = 0; idx < n; idx++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
      if (ArrowArrayAppendInt(locs.get(), -1)) {
        throw std::runtime_error("failed to append int!");
      }
    } else {
      const auto current_size = static_cast<int64_t>(first_occurances.size());

      // TODO: we can make this generic and combine branches if we defined a
      // hashing and comparison operator for the ArrowString types
      if constexpr (std::is_same_v<T, BoolArray> ||
                    std::is_same_v<T, Int64Array>) {
        const auto value =
            ArrowArrayViewGetIntUnsafe(self.array_view_.get(), idx);
        auto did_insert = first_occurances.try_emplace(value, current_size);
        if (did_insert.second) {
          if (ArrowArrayAppendInt(locs.get(), current_size)) {
            throw std::runtime_error("failed to append int!");
          }
          if (ArrowArrayAppendInt(values.get(), value)) {
            throw std::runtime_error("failed to append string");
          }
        } else {
          const int64_t existing_loc = did_insert.first->second;
          if (ArrowArrayAppendInt(locs.get(), existing_loc)) {
            throw std::runtime_error("failed to append int!");
          }
        }
      } else if constexpr (std::is_same_v<T, StringArray>) {
        const auto sv =
            ArrowArrayViewGetStringUnsafe(self.array_view_.get(), idx);
        const std::string_view value{sv.data,
                                     static_cast<size_t>(sv.size_bytes)};

        auto did_insert = first_occurances.try_emplace(value, current_size);
        if (did_insert.second) {
          if (ArrowArrayAppendInt(locs.get(), current_size)) {
            throw std::runtime_error("failed to append int!");
          }
          if (ArrowArrayAppendString(values.get(), sv)) {
            throw std::runtime_error("failed to append string");
          }
        } else {
          const int64_t existing_loc = did_insert.first->second;
          if (ArrowArrayAppendInt(locs.get(), existing_loc)) {
            throw std::runtime_error("failed to append int!");
          }
        }
      } else {
        // see https://stackoverflow.com/a/64354296/621736
        static_assert(!sizeof(T), "Factorize not implemented for type");
      }
    }
  }

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(values.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }
  if (ArrowArrayFinishBuildingDefault(locs.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return std::make_tuple(Int64Array{std::move(locs)}, T{std::move(values)});
}

template <typename T>
std::vector<std::optional<typename T::ScalarT>> ToPyList(const T &self) {
  const auto n = self.array_view_->length;
  std::vector<std::optional<typename T::ScalarT>> result;

  result.reserve(n);
  for (int64_t i = 0; i < n; i++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), i)) {
      result.push_back(std::nullopt);
    } else {
      if constexpr (std::is_same_v<T, BoolArray> ||
                    std::is_same_v<T, Int64Array>) {
        result.push_back(ArrowArrayViewGetIntUnsafe(self.array_view_.get(), i));
      } else if constexpr (std::is_same_v<T, StringArray>) {
        const auto sv =
            ArrowArrayViewGetStringUnsafe(self.array_view_.get(), i);
        const std::string_view value{sv.data,
                                     static_cast<size_t>(sv.size_bytes)};
        result.push_back(value);
      } else {
        // see https://stackoverflow.com/a/64354296/621736
        static_assert(!sizeof(T), "to_pylist not implemented for type");
      }
    }
  }

  return result;
}

template <typename T> T ConcatSameType(const T &self, const T &other) {
  // this implementation assumes that you have a validity and a
  // data buffer, but that is not enforced by compiler yet
  // see also specializations in generic.cpp
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), T::ArrowT)) {
    throw std::runtime_error(
        "Unable to init output array for _concat_same_type!");
  }

  // TODO: check for overflow
  const auto n = self.array_view_->length + other.array_view_->length;

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), n)) {
    throw std::runtime_error("Unable to reserve array!");
  }

  // TODO: would be great to have a more efficient memcpy based algorithm
  // to combine the bitmasks
  for (int64_t idx = 0; idx < self.array_view_.get()->length; idx++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
      if (ArrowArrayAppendNull(result.get(), 1)) {
        throw std::runtime_error("failed to append null!");
      }
    } else {
      if (ArrowArrayAppendEmpty(result.get(), 1)) {
        throw std::runtime_error("failed to append valid!");
      }
    }
  }

  for (int64_t idx = 0; idx < other.array_view_.get()->length; idx++) {
    if (ArrowArrayViewIsNull(other.array_view_.get(), idx)) {
      if (ArrowArrayAppendNull(result.get(), 1)) {
        throw std::runtime_error("failed to append null!");
      }
    } else {
      if (ArrowArrayAppendEmpty(result.get(), 1)) {
        throw std::runtime_error("failed to append valid!");
      }
    }
  }

  const struct ArrowBufferView left_buffer =
      self.array_view_.get()->buffer_views[1];
  const struct ArrowBufferView right_buffer =
      other.array_view_.get()->buffer_views[1];

  const struct ArrowBuffer *result_buffer =
      ArrowArrayBuffer(const_cast<struct ArrowArray *>(result.get()), 1);

  const int64_t left_nbytes = left_buffer.size_bytes;
  const int64_t right_nbytes = right_buffer.size_bytes;

  std::memcpy(result_buffer->data, left_buffer.data.as_uint8, left_nbytes);
  std::memcpy(result_buffer->data + left_nbytes, right_buffer.data.as_uint8,
              right_nbytes);

  struct ArrowError error;
  if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
    throw std::runtime_error("Failed to finish building: " +
                             std::string(error.message));
  }

  return T(std::move(result));
}

template <>
BoolArray ConcatSameType(const BoolArray &self, const BoolArray &other);

template <>
StringArray ConcatSameType(const StringArray &self, const StringArray &other);
