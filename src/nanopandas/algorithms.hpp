#include "array_types.hpp"
#include <nanobind/nanobind.h>
#include <vector>

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
    int64_t value;
    memcpy(&value, cursor, sizeof(uint8_t));
    value = ~value;
    memcpy(cursor, &value, sizeof(uint8_t));
    cursor++;
  }

  return 0;
}

template <typename T> T FromSequence(nb::sequence sequence) { return T(); }

template <typename T>
T FromFactorized(const Int64Array &locs, const T &values) {
  return T();
}

template <typename T>
std::optional<typename T::ScalarT> __getitem__(T &&self, int64_t i) {
  if (i < 0) {
    throw std::range_error("Only positive indexes are supported for now!");
  }

  if (ArrowArrayViewIsNull(self.array_view_.get(), i)) {
    return std::nullopt;
  }

  if constexpr (std::is_same_v<T, BoolArray> || std::is_same_v<T, Int64Array>) {
    return ArrowArrayViewGetIntUnsafe(self.array_view_.get(), i);
  } else if constexpr (std::is_same_v<T, StringArray>) {
    const auto value = ArrowArrayViewGetStringUnsafe(self.array_view_.get(), i);
    return std::string{value.data, static_cast<size_t>(value.size_bytes)};
  } else {
    // see https://stackoverflow.com/a/64354296/621736
    static_assert(!sizeof(T), "__getitem__ not implemented for type");
  }
}

template <typename T> BoolArray __eq__(T &&self, const T &other) {
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

template <typename T> std::string __repr__(T &&self) {}

template <typename T> int64_t __len__(T &&self) {
  return self.array_view_->length;
}

template <typename T> const char *dtype(T &&self) = delete;

template <> const char *dtype(StringArray &&self) { return "string[arrow]"; }

template <> const char *dtype(BoolArray &&self) { return "boolean[arrow]"; }

template <> const char *dtype(Int64Array &&self) { return "int64[arrow]"; }

template <typename T> int64_t nbytes(T &&self) {
  struct ArrowBuffer *data_buffer = ArrowArrayBuffer(
      const_cast<struct ArrowArray *>(self.array_view_.get()->array), 1);
  return data_buffer->size_bytes;
}

template <typename T> int64_t size(T &&self) {
  return self.array_view_->length;
}

template <typename T> bool any(T &&self) {
  return self.array_view_->length > self.array_view_->null_count;
}

template <typename T> bool all(T &&self) {
  return self.array_view_->null_count == 0;
}

template <typename T> BoolArray isna(T &&self) {
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_BOOL)) {
    throw std::runtime_error("Unable to init bool array!");
  }
  const auto n = self.array_view_->length;
  const int64_t bytes_required = _ArrowBytesForBits(n);
  struct ArrowBuffer *buffer = ArrowArrayBuffer(result.get(), 1);
  if (ArrowBufferReserve(buffer, bytes_required)) {
    throw std::runtime_error("Could not reserve arrow buffer");
  }

  ArrowBufferAppendUnsafe(
      buffer, self.array_view_->buffer_views[0].data.as_uint8, bytes_required);
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

template <typename T> T take(T &&self, const std::vector<int64_t> &indices) {
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

  for (const auto idx : indices) {
    if (idx < 0) {
      throw std::range_error("negative indices are not yet implemented");
    } else if (idx > self.array_view_.get()->length) {
      throw std::range_error("index out of bounds!");
    } else {
      if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
        if (ArrowArrayAppendNull(result.get(), 1)) {
          throw std::runtime_error("failed to append null!");
        }
      } else {
        if constexpr (std::is_same_v<T, BoolArray> ||
                      std::is_same_v<T, Int64Array>) {
          const auto value =
              ArrowArrayViewGetIntUnsafe(self.array_view_.get(), idx);
          if (ArrowArrayAppendInt(result.get(), value)) {
            throw std::runtime_error("failed to append int!");
          }
        } else if constexpr (std::is_same_v<T, StringArray>) {
          const auto sv =
              ArrowArrayViewGetStringUnsafe(self.array_view_.get(), idx);
          if (ArrowArrayAppendString(result.get(), sv)) {
            throw std::runtime_error("failed to append string!");
          }
        } else {
          // see https://stackoverflow.com/a/64354296/621736
          static_assert(!sizeof(T), "take not implemented for type");
        }
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

template <typename T> T copy(T &&self) {
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
      if constexpr (std::is_same_v<T, BoolArray> ||
                    std::is_same_v<T, Int64Array>) {
        const auto value =
            ArrowArrayViewGetIntUnsafe(self.array_view_.get(), idx);
        if (ArrowArrayAppendInt(result.get(), value)) {
          throw std::runtime_error("failed to append int!");
        }
      } else if constexpr (std::is_same_v<T, StringArray>) {
        const auto sv =
            ArrowArrayViewGetStringUnsafe(self.array_view_.get(), idx);
        if (ArrowArrayAppendString(result.get(), sv)) {
          throw std::runtime_error("failed to append string!");
        }
      } else {
        // see https://stackoverflow.com/a/64354296/621736
        static_assert(!sizeof(T), "copy not implemented for type");
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

template <typename T> T fillna(T &&self, typename T::ScalarT replacement) {
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

template <typename T> T dropna(T &&self) {
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
      if constexpr (std::is_same_v<T, BoolArray> ||
                    std::is_same_v<T, Int64Array>) {
        const auto value =
            ArrowArrayViewGetIntUnsafe(self.array_view_.get(), idx);
        if (ArrowArrayAppendInt(result.get(), value)) {
          throw std::runtime_error("failed to append int!");
        }
      } else if constexpr (std::is_same_v<T, StringArray>) {
        const auto value =
            ArrowArrayViewGetStringUnsafe(self.array_view_.get(), idx);
        if (ArrowArrayAppendString(result.get(), value)) {
          throw std::runtime_error("failed to append string!");
        }
      } else {
        // see https://stackoverflow.com/a/64354296/621736
        static_assert(!sizeof(T), "dropna not implemented for type");
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

template <typename T> Int64Array len(T &&self) = delete;

template <typename T>
std::vector<std::optional<typename T::ScalarT>> to_pylist(T &&self) {
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
