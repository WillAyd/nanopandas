#pragma once

#include "../array_types.hpp"

#include <utf8proc.h>

StringArray Lower(const StringArray &self);
StringArray Upper(const StringArray &self);
StringArray Capitalize(const StringArray &self);
BoolArray IsAlnum(const StringArray &self);
BoolArray IsAlpha(const StringArray &self);
BoolArray IsDigit(const StringArray &self);
BoolArray IsSpace(const StringArray &self);
BoolArray IsLower(const StringArray &self);
BoolArray IsUpper(const StringArray &self);

template <typename T> Int64Array Len(const T &self) {

  // maybe in the future this could be generically used for containers too
  static_assert(std::is_same_v<T, StringArray>,
                "len is only implemented for StringArray");
  nanoarrow::UniqueArray result;
  if (ArrowArrayInitFromType(result.get(), NANOARROW_TYPE_INT64)) {
    throw std::runtime_error("Unable to init int64 array!");
  }
  const auto n = self.array_view_->length;

  if (ArrowArrayStartAppending(result.get())) {
    throw std::runtime_error("Could not start appending");
  }

  if (ArrowArrayReserve(result.get(), n)) {
    throw std::runtime_error("Unable to reserve array!");
  }

  for (int64_t i = 0; i < n; i++) {
    if (ArrowArrayViewIsNull(self.array_view_.get(), i)) {
      if (ArrowArrayAppendNull(result.get(), 1)) {
        throw std::runtime_error("failed to append null!");
      }
    } else {
      const auto sv = ArrowArrayViewGetStringUnsafe(self.array_view_.get(), i);

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
