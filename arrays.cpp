#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <set>

#include <nanoarrow/nanoarrow_types.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <sstream>
#include <utf8proc.h>

#include <nanoarrow/nanoarrow.hpp>

static_assert(std::is_same_v<std::uint8_t, char> ||
                  std::is_same_v<std::uint8_t, unsigned char>,
              "uint8_t must be a typedef for char or unsigned char");

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

class BoolArray {
public:
  template <typename C> explicit BoolArray(const C &booleans) {
    // TODO: assert we only get bool or std::optional<bool>
    // static_assert(std::is_integral<typename C::value_type>::value ||
    //              std::is_same<typename C::value_type,
    //                           std::optional<bool>>::value);

    if (ArrowArrayInitFromType(array_.get(), NANOARROW_TYPE_BOOL)) {
      throw std::runtime_error("Unable to init BoolArray!");
    };

    if (ArrowArrayStartAppending(array_.get())) {
      throw std::runtime_error("Could not append to BoolArray!");
    }

    for (const auto &opt_boolean : booleans) {
      if (const auto &boolean = opt_boolean) {
        if (ArrowArrayAppendInt(array_.get(), *boolean)) {
          throw std::invalid_argument("Could not append integer: " +
                                      std::to_string(*boolean));
        }
      } else {
        if (ArrowArrayAppendNull(array_.get(), 1)) {
          throw std::invalid_argument("Failed to append null!");
        }
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(array_.get(), &error)) {
      throw std::runtime_error("Failed to finish building array!" +
                               std::string(error.message));
    }

    ArrowArrayViewInitFromType(array_view_.get(), NANOARROW_TYPE_INT64);
    if (ArrowArrayViewSetArray(array_view_.get(), array_.get(), &error)) {
      throw std::runtime_error("Failed to set array view!" +
                               std::string(error.message));
    }
  }

  BoolArray(nanoarrow::UniqueArray &&array) : array_(std::move(array)) {
    ArrowArrayViewInitFromType(array_view_.get(), NANOARROW_TYPE_BOOL);
    struct ArrowError error;
    if (ArrowArrayViewSetArray(array_view_.get(), array_.get(), &error)) {
      throw std::runtime_error("Failed to set array view:" +
                               std::string(error.message));
    }
  }

  // not copyable
  BoolArray(const BoolArray &rhs) = delete;

  int64_t __len__() const { return array_->length; }

  // moving should take ownership of the underlying array
  // TODO: do we need to move array_view?
  BoolArray(BoolArray &&rhs) : BoolArray(std::move(rhs.array_)) {}

  BoolArray &operator=(BoolArray &&rhs) {
    this->array_ = std::move(rhs.array_);
    this->array_view_ = std::move(rhs.array_view_);
    return *this;
  }

  std::vector<std::optional<bool>> to_pylist() {
    const auto n = array_->length;

    std::vector<std::optional<bool>> result;
    result.reserve(n);
    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        result.push_back(std::nullopt);
      } else {
        const auto value = ArrowArrayViewGetIntUnsafe(array_view_.get(), i);
        result.push_back(value);
      }
    }

    return result;
  }

  nanoarrow::UniqueArrayView array_view_;

private:
  nanoarrow::UniqueArray array_;
};

class Int64Array {
public:
  template <typename C> explicit Int64Array(const C &integers) {
    // TODO: assert we only get integral or std::optional<integral>
    // static_assert(std::is_integral<typename C::value_type>::value ||
    //              std::is_same<typename C::value_type,
    //                           std::optional<std::string_view>>::value);

    if (ArrowArrayInitFromType(array_.get(), NANOARROW_TYPE_INT64)) {
      throw std::runtime_error("Unable to init Int64Array!");
    };

    if (ArrowArrayStartAppending(array_.get())) {
      throw std::runtime_error("Could not append to Int64Array!");
    }

    for (const auto &opt_integer : integers) {
      if (const auto &integer = opt_integer) {
        if (ArrowArrayAppendInt(array_.get(), *integer)) {
          throw std::invalid_argument("Could not append integer: " +
                                      std::to_string(*integer));
        }
      } else {
        if (ArrowArrayAppendNull(array_.get(), 1)) {
          throw std::invalid_argument("Failed to append null!");
        }
      }
    }

    if (ArrowArrayFinishBuildingDefault(array_.get(), nullptr)) {
      throw std::runtime_error("Failed to finish building array!");
    }

    ArrowArrayViewInitFromType(array_view_.get(), NANOARROW_TYPE_INT64);
    if (ArrowArrayViewSetArray(array_view_.get(), array_.get(), nullptr)) {
      throw std::runtime_error("Failed to set array view!");
    }
  }

  Int64Array(nanoarrow::UniqueArray &&array) : array_(std::move(array)) {
    ArrowArrayViewInitFromType(array_view_.get(), NANOARROW_TYPE_INT64);
    struct ArrowError error;
    if (ArrowArrayViewSetArray(array_view_.get(), array_.get(), &error)) {
      throw std::runtime_error("Failed to set array view:" +
                               std::string(error.message));
    }
  }

  // not copyable
  Int64Array(const Int64Array &rhs) = delete;

  // moving should take ownership of the underlying array
  // TODO: do we need to move array_view?
  Int64Array(Int64Array &&rhs) : Int64Array(std::move(rhs.array_)) {}

  int64_t __len__() const { return array_->length; }

  Int64Array &operator=(Int64Array &&rhs) {
    this->array_ = std::move(rhs.array_);
    this->array_view_ = std::move(rhs.array_view_);
    return *this;
  }

  std::optional<int64_t> sum() {
    const auto n = array_->length;
    if (array_->length == 0) {
      return std::nullopt;
    }

    int64_t result = 0;
    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        continue;
      } else {
        result += ArrowArrayViewGetIntUnsafe(array_view_.get(), i);
      }
    }

    return result;
  }

  std::optional<int64_t> min() {
    const auto n = array_->length;
    if (array_->length == 0) {
      return std::nullopt;
    }

    int64_t result = std::numeric_limits<int64_t>::max();
    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        continue;
      } else {
        const int64_t value = ArrowArrayViewGetIntUnsafe(array_view_.get(), i);
        if (value < result) {
          result = value;
        }
      }
    }

    return result;
  }

  std::optional<int64_t> max() {
    const auto n = array_->length;
    if (array_->length == 0) {
      return std::nullopt;
    }

    int64_t result = std::numeric_limits<int64_t>::min();
    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        continue;
      } else {
        const int64_t value = ArrowArrayViewGetIntUnsafe(array_view_.get(), i);
        if (value > result) {
          result = value;
        }
      }
    }

    return result;
  }

  std::vector<std::optional<std::int64_t>> to_pylist() {
    const auto n = array_->length;

    std::vector<std::optional<int64_t>> result;
    result.reserve(n);
    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        result.push_back(std::nullopt);
      } else {
        const auto value = ArrowArrayViewGetIntUnsafe(array_view_.get(), i);
        result.push_back(value);
      }
    }

    return result;
  }

  nanoarrow::UniqueArrayView array_view_;

private:
  nanoarrow::UniqueArray array_;
};

class StringArray {
public:
  template <typename C> explicit StringArray(const C &strings) {
    static_assert(std::is_same<typename C::value_type,
                               std::optional<std::string>>::value ||
                  std::is_same<typename C::value_type,
                               std::optional<std::string_view>>::value);
    if (ArrowArrayInitFromType(array_.get(), NANOARROW_TYPE_LARGE_STRING)) {
      throw std::runtime_error("Unable to init StringArray!");
    };

    if (ArrowArrayStartAppending(array_.get())) {
      throw std::runtime_error("Could not append to StringArray!");
    }

    if (ArrowArrayReserve(array_.get(), strings.size())) {
      throw std::runtime_error("Unable to reserve array!");
    }

    for (const auto &opt_str : strings) {
      if (const auto &str = opt_str) {
        struct ArrowStringView sv = {str->data(),
                                     static_cast<int64_t>(str->size())};
        if (ArrowArrayAppendString(array_.get(), sv)) {
          throw std::invalid_argument("Could not append string: " +
                                      std::string(*str));
        }
      } else {
        if (ArrowArrayAppendNull(array_.get(), 1)) {
          throw std::invalid_argument("Failed to append null!");
        }
      }
    }

    if (ArrowArrayFinishBuildingDefault(array_.get(), nullptr)) {
      throw std::runtime_error("Failed to finish building array!");
    }

    ArrowArrayViewInitFromType(array_view_.get(), NANOARROW_TYPE_LARGE_STRING);
    if (ArrowArrayViewSetArray(array_view_.get(), array_.get(), nullptr)) {
      throw std::runtime_error("Failed to set array view!");
    }
  }

  StringArray(nanoarrow::UniqueArray &&array) : array_(std::move(array)) {
    ArrowArrayViewInitFromType(array_view_.get(), NANOARROW_TYPE_LARGE_STRING);
    struct ArrowError error;
    if (ArrowArrayViewSetArray(array_view_.get(), array_.get(), &error)) {
      throw std::runtime_error("Failed to set array view:" +
                               std::string(error.message));
    }
  }

  // not copyable
  StringArray(const StringArray &rhs) = delete;
  // moving should take ownership of the underlying array
  // TODO: do we need to move array_view?
  StringArray(StringArray &&rhs) : StringArray(std::move(rhs.array_)) {}

  StringArray _from_sequence(nb::sequence sequence) {
    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_LARGE_STRING)) {
      throw std::runtime_error("Unable to init large string array!");
    }
    const auto n = array_->length;

    if (ArrowArrayStartAppending(result.get())) {
      throw std::runtime_error("Could not start appending");
    }

    for (const auto &item : sequence) {
      if (item.is_none()) {
        if (ArrowArrayAppendNull(result.get(), 1)) {
          throw std::runtime_error("failed to append null!");
        }
      } else {
        std::string_view sv = nb::cast<std::string_view>(item);
        const struct ArrowStringView arrow_sv = {
            sv.data(), static_cast<int64_t>(sv.size())};
        if (ArrowArrayAppendString(result.get(), arrow_sv)) {
          throw std::runtime_error("failed to append string!");
        }
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    return StringArray(std::move(result));
  }

  StringArray _from_factorized(const Int64Array &locs,
                               const StringArray &strings) {
    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_LARGE_STRING)) {
      throw std::runtime_error("Unable to init large string array!");
    }
    const auto n = locs.__len__();

    if (ArrowArrayStartAppending(result.get())) {
      throw std::runtime_error("Could not append to StringArray!");
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
        const auto sv =
            ArrowArrayViewGetStringUnsafe(strings.array_view_.get(), loc_value);
        if (ArrowArrayAppendString(result.get(), sv)) {
          throw std::runtime_error("failed to append string!");
        }
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    return StringArray(std::move(result));
  }

  std::optional<std::string> __getitem__(int64_t i) {
    if (i < 0) {
      throw std::range_error("Only positive indexes are supported for now!");
    }

    if (ArrowArrayViewIsNull(array_view_.get(), i)) {
      return std::nullopt;
    } else {
      const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);
      return std::string{sv.data, static_cast<size_t>(sv.size_bytes)};
    }
  }

  int64_t __len__() const { return array_->length; }

  BoolArray __eq__(const StringArray &other) {
    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_BOOL)) {
      throw std::runtime_error("Unable to init bool array!");
    }
    const auto n = array_->length;

    if (n != other.array_->length) {
      throw std::range_error("Arrays are not of equal size");
    }

    if (ArrowArrayStartAppending(result.get())) {
      throw std::runtime_error("Could not start appending");
    }

    if (ArrowArrayReserve(result.get(), n)) {
      throw std::runtime_error("Unable to reserve array!");
    }

    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        if (ArrowArrayAppendNull(result.get(), 1)) {
          throw std::runtime_error("failed to append null!");
        }
      } else {
        const auto left = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);
        const auto right =
            ArrowArrayViewGetStringUnsafe(other.array_view_.get(), i);
        // TODO: we are doing a byte comparison - do we need a unicode compare
        // instead?
        const auto nbytes = left.size_bytes;
        if ((nbytes == right.size_bytes) &&
            (!strncmp(left.data, right.data, static_cast<size_t>(nbytes)))) {
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

  const char *dtype() { return "string[arrow]"; }

  int64_t nbytes() {
    struct ArrowBuffer *data_buffer = ArrowArrayBuffer(array_.get(), 1);
    return data_buffer->size_bytes;
  }

  BoolArray isna() {
    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_BOOL)) {
      throw std::runtime_error("Unable to init bool array!");
    }
    const auto n = array_->length;
    const int64_t bytes_required = _ArrowBytesForBits(n);
    struct ArrowBuffer *buffer = ArrowArrayBuffer(result.get(), 1);
    if (ArrowBufferReserve(buffer, bytes_required)) {
      throw std::runtime_error("Could not reserve arrow buffer");
    }

    ArrowBufferAppendUnsafe(buffer, array_view_->buffer_views[0].data.as_uint8,
                            bytes_required);
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

  StringArray take(const std::vector<int64_t> &indices) {
    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_LARGE_STRING)) {
      throw std::runtime_error("Unable to init large string array!");
    }
    const auto n = array_->length;

    if (ArrowArrayStartAppending(result.get())) {
      throw std::runtime_error("Could not start appending");
    }

    if (ArrowArrayReserve(result.get(), indices.size())) {
      throw std::runtime_error("Unable to reserve array!");
    }

    for (const auto idx : indices) {
      if (idx < 0) {
        throw std::range_error("negative indices are not yet implemented");
      } else if (idx > array_.get()->length) {
        throw std::range_error("index out of bounds!");
      } else {
        if (ArrowArrayViewIsNull(array_view_.get(), idx)) {
          if (ArrowArrayAppendNull(result.get(), 1)) {
            throw std::runtime_error("failed to append null!");
          }
        } else {
          const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), idx);
          if (ArrowArrayAppendString(result.get(), sv)) {
            throw std::runtime_error("failed to append string!");
          }
        }
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    return StringArray(std::move(result));
  }

  StringArray copy() {
    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_LARGE_STRING)) {
      throw std::runtime_error("Unable to init large string array!");
    }
    const auto n = array_->length;

    if (ArrowArrayStartAppending(result.get())) {
      throw std::runtime_error("Could not start appending");
    }

    if (ArrowArrayReserve(result.get(), n)) {
      throw std::runtime_error("Unable to reserve array!");
    }

    for (int64_t idx = 0; idx < n; idx++) {
      if (ArrowArrayViewIsNull(array_view_.get(), idx)) {
        if (ArrowArrayAppendNull(result.get(), 1)) {
          throw std::runtime_error("failed to append null!");
        }
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), idx);
        if (ArrowArrayAppendString(result.get(), sv)) {
          throw std::runtime_error("failed to append string!");
        }
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    return StringArray(std::move(result));
  }

  StringArray _concat_same_type(const StringArray &other) {
    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_LARGE_STRING)) {
      throw std::runtime_error("Unable to init large string array!");
    }

    // TODO: check for overflow
    const auto n = array_->length + other.array_->length;

    if (ArrowArrayStartAppending(result.get())) {
      throw std::runtime_error("Could not start appending");
    }

    if (ArrowArrayReserve(result.get(), n)) {
      throw std::runtime_error("Unable to reserve array!");
    }

    for (int64_t idx = 0; idx < array_.get()->length; idx++) {
      if (ArrowArrayViewIsNull(array_view_.get(), idx)) {
        if (ArrowArrayAppendNull(result.get(), 1)) {
          throw std::runtime_error("failed to append null!");
        }
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), idx);
        if (ArrowArrayAppendString(result.get(), sv)) {
          throw std::runtime_error("failed to append string!");
        }
      }
    }

    for (int64_t idx = 0; idx < other.array_->length; idx++) {
      if (ArrowArrayViewIsNull(other.array_view_.get(), idx)) {
        if (ArrowArrayAppendNull(result.get(), 1)) {
          throw std::runtime_error("failed to append null!");
        }
      } else {
        const auto sv =
            ArrowArrayViewGetStringUnsafe(other.array_view_.get(), idx);
        if (ArrowArrayAppendString(result.get(), sv)) {
          throw std::runtime_error("failed to append string!");
        }
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    return StringArray(std::move(result));
  }

  StringArray interpolate() {
    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_LARGE_STRING)) {
      throw std::runtime_error("Unable to init large string array!");
    }

    // TODO: check for overflow
    const auto n = array_->length;

    if (ArrowArrayStartAppending(result.get())) {
      throw std::runtime_error("Could not start appending");
    }

    if (ArrowArrayReserve(result.get(), n)) {
      throw std::runtime_error("Unable to reserve array!");
    }

    bool seen_value = false;
    struct ArrowStringView most_recent_sv;

    for (int64_t idx = 0; idx < array_.get()->length; idx++) {
      if (ArrowArrayViewIsNull(array_view_.get(), idx)) {
        if (!seen_value) {
          if (ArrowArrayAppendNull(result.get(), 1)) {
            throw std::runtime_error("failed to append null!");
          }
        } else {
          if (ArrowArrayAppendString(result.get(), most_recent_sv)) {
            throw std::runtime_error("failed to append string!");
          }
        }
      } else {
        seen_value = true;
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), idx);
        if (ArrowArrayAppendString(result.get(), sv)) {
          throw std::runtime_error("failed to append string!");
        }
        most_recent_sv = sv;
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    return StringArray(std::move(result));
  }

  std::string __repr__() {
    std::ostringstream out{};
    out << "StringArray\n[";
    constexpr size_t maxchars = 40;
    size_t chars_written = 1;
    const auto n = array_->length;
    for (int64_t idx = 0; idx < n; idx++) {
      if (ArrowArrayViewIsNull(array_view_.get(), idx)) {
        out << "null";
        chars_written += 4;
        if (chars_written >= maxchars) {
          break;
        }
      } else {
        out << "\"";
        chars_written += 1;
        if (chars_written >= maxchars) {
          break;
        }
        const auto arrow_sv =
            ArrowArrayViewGetStringUnsafe(array_view_.get(), idx);

        const auto nbytes = static_cast<size_t>(arrow_sv.size_bytes);
        const size_t bytes_to_write = (maxchars - chars_written) > nbytes
                                          ? nbytes
                                          : maxchars - chars_written;

        const std::string_view sv{arrow_sv.data, bytes_to_write};
        out << sv;
        chars_written += bytes_to_write;
        if (chars_written >= maxchars) {
          break;
        }

        out << "\"";
        chars_written += 1;
        if (chars_written >= maxchars) {
          break;
        }
      }
      if (idx < n - 1) {
        out << ", ";
        chars_written += 1;
        if (chars_written >= maxchars) {
          break;
        }

      } else {
        out << "]";
        chars_written += 1;
        if (chars_written >= maxchars) {
          break;
        }
      }
    }

    return out.str();
  }

  StringArray fillna(std::string_view replacement) {
    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_LARGE_STRING)) {
      throw std::runtime_error("Unable to init large string array!");
    }

    const auto n = array_->length;

    if (ArrowArrayStartAppending(result.get())) {
      throw std::runtime_error("Could not start appending");
    }

    if (ArrowArrayReserve(result.get(), n)) {
      throw std::runtime_error("Unable to reserve array!");
    }

    const struct ArrowStringView replacement_sv = {
        replacement.data(), static_cast<int64_t>(replacement.size())};
    for (int64_t idx = 0; idx < array_.get()->length; idx++) {
      if (ArrowArrayViewIsNull(array_view_.get(), idx)) {
        if (ArrowArrayAppendString(result.get(), replacement_sv)) {
          throw std::runtime_error("failed to append string!");
        }
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), idx);
        if (ArrowArrayAppendString(result.get(), sv)) {
          throw std::runtime_error("failed to append string!");
        }
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    return StringArray(std::move(result));
  }

  StringArray _pad_or_backfill(std::string_view method) {
    if ((method != "pad") && (method != "backfill")) {
      throw std::invalid_argument(
          "'method' must be either 'pad' or 'backfill'");
    }

    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_LARGE_STRING)) {
      throw std::runtime_error("Unable to init large string array!");
    }

    const auto n = array_->length;

    if (ArrowArrayStartAppending(result.get())) {
      throw std::runtime_error("Could not start appending");
    }

    if (ArrowArrayReserve(result.get(), n)) {
      throw std::runtime_error("Unable to reserve array!");
    }

    if (method == "pad") {
      bool seen_value = false;
      struct ArrowStringView most_recent_sv;
      for (int64_t idx = 0; idx < array_.get()->length; idx++) {
        if (ArrowArrayViewIsNull(array_view_.get(), idx)) {
          if (!seen_value) {
            if (ArrowArrayAppendNull(result.get(), 1)) {
              throw std::runtime_error("failed to append null!");
            }
          } else {
            if (ArrowArrayAppendString(result.get(), most_recent_sv)) {
              throw std::runtime_error("failed to append string!");
            }
          }
        } else {
          seen_value = true;
          const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), idx);
          if (ArrowArrayAppendString(result.get(), sv)) {
            throw std::runtime_error("failed to append string!");
          }
          most_recent_sv = sv;
        }
      }
    } else {
      struct ArrowStringView next_sv;
      int64_t last_append = 0;
      for (int64_t idx = 0; idx < array_.get()->length; idx++) {
        if (ArrowArrayViewIsNull(array_view_.get(), idx)) {
          if (idx == array_.get()->length - 1) {
            do {
              if (ArrowArrayAppendNull(result.get(), 1)) {
                throw std::runtime_error("failed to append null!");
              }
              last_append++;
            } while (last_append <= idx);
          }
          continue;
        } else {
          const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), idx);
          do {
            if (ArrowArrayAppendString(result.get(), sv)) {
              throw std::runtime_error("failed to append string!");
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

    return StringArray(std::move(result));
  }

  StringArray dropna() {
    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_LARGE_STRING)) {
      throw std::runtime_error("Unable to init large string array!");
    }

    const auto n = array_->length - array_view_->null_count;
    if (ArrowArrayStartAppending(result.get())) {
      throw std::runtime_error("Could not start appending");
    }

    if (ArrowArrayReserve(result.get(), n)) {
      throw std::runtime_error("Unable to reserve array!");
    }

    for (int64_t idx = 0; idx < array_.get()->length; idx++) {
      if (ArrowArrayViewIsNull(array_view_.get(), idx)) {
        continue;
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), idx);
        if (ArrowArrayAppendString(result.get(), sv)) {
          throw std::runtime_error("failed to append string!");
        }
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    return StringArray(std::move(result));
  }

  Int64Array len() {
    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_INT64)) {
      throw std::runtime_error("Unable to init int64 array!");
    }
    const auto n = array_->length;

    if (ArrowArrayStartAppending(result.get())) {
      throw std::runtime_error("Could not start appending");
    }

    if (ArrowArrayReserve(result.get(), n)) {
      throw std::runtime_error("Unable to reserve array!");
    }

    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        if (ArrowArrayAppendNull(result.get(), i)) {
          throw std::runtime_error("failed to append null!");
        }
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);

        size_t niter = 0;
        size_t bytes_read = 0;
        size_t bytes_rem;
        while ((bytes_rem = static_cast<size_t>(sv.size_bytes) - bytes_read) >
               0) {
          utf8proc_int32_t codepoint;
          size_t codepoint_bytes = utf8proc_iterate(
              reinterpret_cast<const utf8proc_uint8_t *>(sv.data + bytes_read),
              bytes_rem, &codepoint);

          niter++;
          bytes_read += codepoint_bytes;
        }

        if (ArrowArrayAppendInt(result.get(), niter)) {
          throw std::runtime_error("failed to append int!");
        }
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    return Int64Array(std::move(result));
  }

  StringArray lower() {
    std::vector<std::optional<std::string>> result;
    const auto n = array_->length;

    result.reserve(n);
    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        result.push_back(std::nullopt);
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);
        unsigned char *dst;

        constexpr auto lambda = [](utf8proc_int32_t codepoint,
                                   void *unused) -> utf8proc_int32_t {
          return utf8proc_tolower(codepoint);
        };

        const ssize_t nbytes = utf8proc_map_custom(
            reinterpret_cast<const uint8_t *>(sv.data), sv.size_bytes, &dst,
            UTF8PROC_STABLE, lambda, NULL);
        if (nbytes < 0) {
          throw std::runtime_error("error occurred converting tolower!");
        }

        // utf8proc and std::string but malloc on the heap; maybe we could avoid
        // a second malloc with a StringArray constructor for raw C pointers
        std::string converted{};
        converted.resize(nbytes);
        memcpy(&converted[0], dst, nbytes);
        free(dst);

        result.push_back(std::move(converted));
      }
    }

    return StringArray{result};
  }

  StringArray upper() {
    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_LARGE_STRING)) {
      throw std::runtime_error("Unable to init large string array!");
    }
    const auto n = array_->length;

    if (ArrowArrayStartAppending(result.get())) {
      throw std::runtime_error("Could not start appending");
    }

    if (ArrowArrayReserve(result.get(), n)) {
      throw std::runtime_error("Unable to reserve array!");
    }

    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        if (ArrowArrayAppendNull(result.get(), 1)) {
          throw std::runtime_error("failed to append null!");
        }
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);
        unsigned char *dst;

        constexpr auto lambda = [](utf8proc_int32_t codepoint,
                                   void *unused) -> utf8proc_int32_t {
          return utf8proc_toupper(codepoint);
        };

        const ssize_t nbytes = utf8proc_map_custom(
            reinterpret_cast<const uint8_t *>(sv.data), sv.size_bytes, &dst,
            UTF8PROC_STABLE, lambda, NULL);
        if (nbytes < 0) {
          throw std::runtime_error("error occurred converting toupper!");
        }

        struct ArrowStringView dest_sv = {reinterpret_cast<const char *>(dst),
                                          nbytes};

        if (ArrowArrayAppendString(result.get(), dest_sv)) {
          throw std::runtime_error("failed to append string");
        }
        free(dst);
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(result.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    return StringArray{std::move(result)};
  }

  StringArray capitalize() {
    std::vector<std::optional<std::string>> result;
    const auto n = array_->length;

    result.reserve(n);
    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        result.push_back(std::nullopt);
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);
        std::vector<utf8proc_uint8_t> dst;
        dst.reserve(static_cast<size_t>(sv.size_bytes));

        size_t bytes_read = 0;
        size_t bytes_rem;
        while ((bytes_rem = static_cast<size_t>(sv.size_bytes) - bytes_read) >
               0) {
          utf8proc_int32_t codepoint;
          size_t codepoint_bytes = utf8proc_iterate(
              reinterpret_cast<const utf8proc_uint8_t *>(sv.data + bytes_read),
              bytes_rem, &codepoint);

          if (bytes_read ==
              0) { // maybe not the best impl, but easier to template with upper
            codepoint = utf8proc_toupper(codepoint);
          }
          bytes_read += codepoint_bytes;

          std::array<utf8proc_uint8_t, 4> encoded;
          utf8proc_encode_char(codepoint, encoded.data());
          dst.insert(dst.end(), encoded.begin(),
                     encoded.begin() + codepoint_bytes);
        }
        result.push_back(std::string{dst.begin(), dst.end()});
      }
    }

    return StringArray{result};
  }

  BoolArray isalnum() {
    constexpr auto lambda = [](utf8proc_int32_t codepoint) {
      const auto category = utf8proc_category(codepoint);
      switch (category) {
      case UTF8PROC_CATEGORY_LU:
      case UTF8PROC_CATEGORY_LL:
      case UTF8PROC_CATEGORY_LT:
      case UTF8PROC_CATEGORY_LM:
      case UTF8PROC_CATEGORY_LO:
      case UTF8PROC_CATEGORY_ND:
      case UTF8PROC_CATEGORY_NL:
      case UTF8PROC_CATEGORY_NO:
        return true;
      default:
        return false;
      }
    };

    return IsFuncApplicator(lambda);
  }

  BoolArray isalpha() {
    constexpr auto lambda = [](utf8proc_int32_t codepoint) {
      const auto category = utf8proc_category(codepoint);
      switch (category) {
      case UTF8PROC_CATEGORY_LU:
      case UTF8PROC_CATEGORY_LL:
      case UTF8PROC_CATEGORY_LT:
      case UTF8PROC_CATEGORY_LM:
      case UTF8PROC_CATEGORY_LO:
        return true;
      default:
        return false;
      }
    };

    return IsFuncApplicator(lambda);
  }

  BoolArray isdigit() {
    constexpr auto lambda = [](utf8proc_int32_t codepoint) {
      const auto category = utf8proc_category(codepoint);
      switch (category) {
      case UTF8PROC_CATEGORY_ND:
      case UTF8PROC_CATEGORY_NL:
      case UTF8PROC_CATEGORY_NO:
        return true;
      default:
        return false;
      }
    };

    return IsFuncApplicator(lambda);
  }

  BoolArray isspace() {
    constexpr auto lambda = [](utf8proc_int32_t codepoint) {
      const auto category = utf8proc_category(codepoint);
      switch (category) {
      case UTF8PROC_CATEGORY_ZS:
      case UTF8PROC_CATEGORY_ZL:
      case UTF8PROC_CATEGORY_ZP:
        return true;
      default:
        return false;
      }
    };

    return IsFuncApplicator(lambda);
  }

  BoolArray islower() {
    constexpr auto lambda = [](utf8proc_int32_t codepoint) {
      return utf8proc_islower(codepoint);
    };

    return IsFuncApplicator(lambda);
  }

  BoolArray isupper() {
    constexpr auto lambda = [](utf8proc_int32_t codepoint) {
      return utf8proc_isupper(codepoint);
    };

    return IsFuncApplicator(lambda);
  }

  // Misc extras
  int64_t size() { return array_->length; }

  bool any() { return array_->length > array_->null_count; }
  bool all() { return array_->null_count == 0; }

  StringArray unique() {
    // TODO: this should never be optional in unique; simply required by current
    // constructor but there is probably a smarter way to template that
    std::set<std::optional<std::string>> result;
    const auto n = array_->length;

    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        continue;
      }

      const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);
      const std::string value{sv.data, static_cast<size_t>(sv.size_bytes)};
      result.insert(value);
    }

    return StringArray{result};
  }

  std::tuple<Int64Array, StringArray> factorize() {
    constexpr auto hashfunc = [](struct ArrowStringView sv) -> size_t {
      const auto cppsv =
          std::string_view{sv.data, static_cast<size_t>(sv.size_bytes)};
      return std::hash<int64_t>()(sv.size_bytes) ^
             std::hash<std::string_view>()(cppsv);
    };

    constexpr auto comp = [](struct ArrowStringView sv1,
                             struct ArrowStringView sv2) -> bool {
      const int64_t nbytes = sv1.size_bytes;
      if (nbytes != sv2.size_bytes) {
        return false;
      }

      return !strncmp(sv1.data, sv2.data, static_cast<size_t>(nbytes));
    };

    std::unordered_map<struct ArrowStringView, int64_t, decltype(hashfunc),
                       decltype(comp)>
        first_occurances{0, hashfunc, comp};

    nanoarrow::UniqueArray strings;
    if (ArrowArrayInitFromType(strings.get(), NANOARROW_TYPE_LARGE_STRING)) {
      throw std::runtime_error("Unable to init string array!");
    }
    nanoarrow::UniqueArray locs;
    if (ArrowArrayInitFromType(locs.get(), NANOARROW_TYPE_INT64)) {
      throw std::runtime_error("Unable to init int64 array!");
    }
    const auto n = array_->length;

    if (ArrowArrayStartAppending(strings.get())) {
      throw std::runtime_error("Could not start appending");
    }

    if (ArrowArrayStartAppending(locs.get())) {
      throw std::runtime_error("Could not start appending");
    }

    for (int64_t idx = 0; idx < n; idx++) {
      if (ArrowArrayViewIsNull(array_view_.get(), idx)) {
        if (ArrowArrayAppendInt(locs.get(), -1)) {
          throw std::runtime_error("failed to append int!");
        }
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), idx);

        const auto current_size = static_cast<int64_t>(first_occurances.size());
        auto did_insert = first_occurances.try_emplace(sv, current_size);
        if (did_insert.second) {
          if (ArrowArrayAppendInt(locs.get(), current_size)) {
            throw std::runtime_error("failed to append int!");
          }
          if (ArrowArrayAppendString(strings.get(), sv)) {
            throw std::runtime_error("failed to append string");
          }
        } else {
          const int64_t existing_loc = did_insert.first->second;
          if (ArrowArrayAppendInt(locs.get(), existing_loc)) {
            throw std::runtime_error("failed to append int!");
          }
        }
      }
    }

    struct ArrowError error;
    if (ArrowArrayFinishBuildingDefault(strings.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }
    if (ArrowArrayFinishBuildingDefault(locs.get(), &error)) {
      throw std::runtime_error("Failed to finish building: " +
                               std::string(error.message));
    }

    return std::make_tuple(Int64Array{std::move(locs)},
                           StringArray{std::move(strings)});
  }

  std::vector<std::optional<std::string_view>> to_pylist() {
    const auto n = array_->length;

    std::vector<std::optional<std::string_view>> result;
    result.reserve(n);
    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        result.push_back(std::nullopt);
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);
        const std::string_view value{sv.data,
                                     static_cast<size_t>(sv.size_bytes)};
        result.push_back(value);
      }
    }

    return result;
  }

  nanoarrow::UniqueArrayView array_view_;

private:
  BoolArray
  IsFuncApplicator(const std::function<bool(utf8proc_int32_t)> &lambda) {
    nanoarrow::UniqueArray result;
    if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_BOOL)) {
      throw std::runtime_error("Unable to init bool array!");
    }
    const auto n = array_->length;

    if (ArrowArrayStartAppending(result.get())) {
      throw std::runtime_error("Could not start appending");
    }

    if (ArrowArrayReserve(result.get(), n)) {
      throw std::runtime_error("Unable to reserve array!");
    }

    for (int64_t i = 0; i < n; i++) {
      if (ArrowArrayViewIsNull(array_view_.get(), i)) {
        if (ArrowArrayAppendNull(result.get(), i)) {
          throw std::runtime_error("failed to append null!");
        }
      } else {
        const auto sv = ArrowArrayViewGetStringUnsafe(array_view_.get(), i);

        size_t bytes_read = 0;
        size_t bytes_rem;

        bool is_true = true;
        while ((bytes_rem = static_cast<size_t>(sv.size_bytes) - bytes_read) >
               0) {
          utf8proc_int32_t codepoint;
          size_t codepoint_bytes = utf8proc_iterate(
              reinterpret_cast<const utf8proc_uint8_t *>(sv.data + bytes_read),
              bytes_rem, &codepoint);

          is_true = lambda(codepoint);
          if (!is_true) {
            break;
          }

          bytes_read += codepoint_bytes;
        }

        if (ArrowArrayAppendInt(result.get(), is_true)) {
          throw std::runtime_error("failed to append bool!");
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

  nanoarrow::UniqueArray array_;
};

// try to match all pandas methods
// https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#method-summary
NB_MODULE(nanopandas, m) {
  nb::class_<BoolArray>(m, "BoolArray")
      .def(nb::init<std::vector<std::optional<int64_t>>>())
      .def("__len__", &BoolArray::__len__)
      .def("to_pylist", &BoolArray::to_pylist);

  nb::class_<Int64Array>(m, "Int64Array")
      .def(nb::init<std::vector<std::optional<int64_t>>>())
      .def("__len__", &Int64Array::__len__)
      .def("sum", &Int64Array::sum)
      .def("min", &Int64Array::min)
      .def("max", &Int64Array::max)
      .def("to_pylist", &Int64Array::to_pylist);

  nb::class_<StringArray>(m, "StringArray")
      .def(nb::init<std::vector<std::optional<std::string_view>>>())

      // extension array interface
      .def("_from_sequence", &StringArray::_from_sequence)
      .def("_from_factorized", &StringArray::_from_factorized)
      .def("__getitem__", &StringArray::__getitem__)
      .def("__len__", &StringArray::__len__)
      .def("__eq__", &StringArray::__eq__)
      .def_prop_ro("dtype", &StringArray::dtype)
      .def_prop_ro("nbytes", &StringArray::nbytes)
      .def("isna", &StringArray::isna)
      .def("take", &StringArray::take)
      .def("copy", &StringArray::copy)
      .def("_concat_same_type", &StringArray::_concat_same_type)
      .def("interpolate", &StringArray::interpolate)

      // formatting methods
      .def("__repr__", &StringArray::__repr__)

      // extra interface methods
      .def("fillna", &StringArray::fillna)
      .def("_pad_or_backfill", &StringArray::_pad_or_backfill)
      .def("dropna", &StringArray::dropna)
      .def("unique", &StringArray::unique)
      .def("factorize", &StringArray::factorize)

      // str accessor methods
      .def("len", &StringArray::len)
      .def("lower", &StringArray::lower)
      .def("upper", &StringArray::upper)
      .def("capitalize", &StringArray::capitalize)
      .def("isalnum", &StringArray::isalnum)
      .def("isalpha", &StringArray::isalpha)
      .def("isdigit", &StringArray::isdigit)
      .def("isspace", &StringArray::isspace)
      .def("islower", &StringArray::islower)
      .def("isupper", &StringArray::isupper)
      // some extras that may be useful
      .def_prop_ro("size", &StringArray::size)

      .def("any", &StringArray::any)
      .def("all", &StringArray::all)
      .def("to_pylist", &StringArray::to_pylist);
}
