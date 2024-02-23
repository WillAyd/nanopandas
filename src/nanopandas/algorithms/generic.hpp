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
    int64_t value;
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
      if constexpr (std::is_same_v<T, BoolArray> ||
                    std::is_same_v<T, Int64Array>) {
        const auto value =
            ArrowArrayViewGetIntUnsafe(values.array_view_.get(), loc_value);
        if (ArrowArrayAppendInt(result.get(), value)) {
          throw std::runtime_error("failed to append int!");
        }
      } else if constexpr (std::is_same_v<T, StringArray>) {
        const auto value =
            ArrowArrayViewGetStringUnsafe(values.array_view_.get(), loc_value);
        if (ArrowArrayAppendString(result.get(), value)) {
          throw std::runtime_error("failed to append string!");
        }
      } else {
        // see https://stackoverflow.com/a/64354296/621736
        static_assert(!sizeof(T), "FromFactorized not implemented for type");
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
std::optional<typename T::ScalarT> GetItemDunder(const T &self, int64_t i) {
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
    return
        typename T::ScalarT{value.data, static_cast<size_t>(value.size_bytes)};
  } else {
    // see https://stackoverflow.com/a/64354296/621736
    static_assert(!sizeof(T), "__getitem__ not implemented for type");
  }
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

template <typename T> const char *Dtype([[maybe_unused]] const T &self);

template <typename T> int64_t Nbytes(const T &self) {
  const struct ArrowBufferView data_buffer =
      self.array_view_.get()->buffer_views[1];
  return data_buffer.size_bytes;
}

template <typename T> int64_t Size(const T &self) {
  return self.array_view_->length;
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
  typename T::ScalarT last_value_seen;

  for (int64_t idx = 0; idx < self.array_view_.get()->length; idx++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
      if (!seen_value) {
        if (ArrowArrayAppendNull(result.get(), 1)) {
          throw std::runtime_error("failed to append null!");
        }
      } else {
        if constexpr (std::is_same_v<T, BoolArray> ||
                      std::is_same_v<T, Int64Array>) {
          if (ArrowArrayAppendInt(result.get(), last_value_seen)) {
            throw std::runtime_error("failed to append int value!");
          }
        } else if constexpr (std::is_same_v<T, StringArray>) {
          const struct ArrowStringView sv {
            last_value_seen.data(), static_cast<int64_t>(last_value_seen.size())
          };
          if (ArrowArrayAppendString(result.get(), sv)) {
            throw std::runtime_error("failed to append string!");
          }
        } else {
          // see https://stackoverflow.com/a/64354296/621736
          static_assert(!sizeof(T), "interpolate not implemented for type");
        }
      }
    } else {
      seen_value = true;

      if constexpr (std::is_same_v<T, BoolArray> ||
                    std::is_same_v<T, Int64Array>) {
        const auto value =
            ArrowArrayViewGetIntUnsafe(self.array_view_.get(), idx);
        if (ArrowArrayAppendInt(result.get(), value)) {
          throw std::runtime_error("failed to append int!");
        }
        last_value_seen = value;
      } else if constexpr (std::is_same_v<T, StringArray>) {
        const auto value =
            ArrowArrayViewGetStringUnsafe(self.array_view_.get(), idx);
        if (ArrowArrayAppendString(result.get(), value)) {
          throw std::runtime_error("failed to append string!");
        }
        last_value_seen =
            std::string_view{value.data, static_cast<size_t>(value.size_bytes)};
      } else {
        // see https://stackoverflow.com/a/64354296/621736
        static_assert(!sizeof(T), "interpolate not implemented for type");
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
    typename T::ScalarT last_value_seen;
    for (int64_t idx = 0; idx < self.array_view_.get()->length; idx++) {
      if (ArrowArrayViewIsNull(self.array_view_.get(), idx)) {
        if (!seen_value) {
          if (ArrowArrayAppendNull(result.get(), 1)) {
            throw std::runtime_error("failed to append null!");
          }
        } else {
          if constexpr (std::is_same_v<T, BoolArray> ||
                        std::is_same_v<T, Int64Array>) {
            if (ArrowArrayAppendInt(result.get(), last_value_seen)) {
              throw std::runtime_error("failed to append int value!");
            }
          } else if constexpr (std::is_same_v<T, StringArray>) {
            const struct ArrowStringView sv {
              last_value_seen.data(),
                  static_cast<int64_t>(last_value_seen.size())
            };
            if (ArrowArrayAppendString(result.get(), sv)) {
              throw std::runtime_error("failed to append string!");
            }
          } else {
            // see https://stackoverflow.com/a/64354296/621736
            static_assert(!sizeof(T), "interpolate not implemented for type");
          }
        }
      } else {
        seen_value = true;

        if constexpr (std::is_same_v<T, BoolArray> ||
                      std::is_same_v<T, Int64Array>) {
          const auto value =
              ArrowArrayViewGetIntUnsafe(self.array_view_.get(), idx);
          if (ArrowArrayAppendInt(result.get(), value)) {
            throw std::runtime_error("failed to append int!");
          }
          last_value_seen = value;
        } else if constexpr (std::is_same_v<T, StringArray>) {
          const auto value =
              ArrowArrayViewGetStringUnsafe(self.array_view_.get(), idx);
          if (ArrowArrayAppendString(result.get(), value)) {
            throw std::runtime_error("failed to append string!");
          }
          last_value_seen = std::string_view{
              value.data, static_cast<size_t>(value.size_bytes)};
        } else {
          // see https://stackoverflow.com/a/64354296/621736
          static_assert(!sizeof(T), "interpolate not implemented for type");
        }
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
        if constexpr (std::is_same_v<T, BoolArray> ||
                      std::is_same_v<T, Int64Array>) {
          const auto value =
              ArrowArrayViewGetIntUnsafe(self.array_view_.get(), idx);
          do {
            if (ArrowArrayAppendInt(result.get(), value)) {
              throw std::runtime_error("failed to append int!");
            }
            last_append++;
          } while (last_append <= idx);
        } else if constexpr (std::is_same_v<T, StringArray>) {
          const auto value =
              ArrowArrayViewGetStringUnsafe(self.array_view_.get(), idx);
          do {
            if (ArrowArrayAppendString(result.get(), value)) {
              throw std::runtime_error("failed to append string!");
            }
            last_append++;
          } while (last_append <= idx);
        } else {
          // see https://stackoverflow.com/a/64354296/621736
          static_assert(!sizeof(T), "PadOrBackfill not implemented for type");
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
