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

template <typename T> nb::object __getitem__(T &&self, int64_t i) {}

template <typename T> BoolArray __eq__(T &&self, const T &other) {}

template <typename T> std::string __repr__(T &&self) {}

template <typename T> int64_t __len__(T &&self) { return self.array_->length; }

template <typename T> const char *dtype(T &&self) = delete;

template <> const char *dtype(StringArray &&self) { return "string[arrow]"; }

template <> const char *dtype(BoolArray &&self) { return "boolean[arrow]"; }

template <> const char *dtype(Int64Array &&self) { return "int64[arrow]"; }

template <typename T> int64_t nbytes(T &&self) {
  struct ArrowBuffer *data_buffer = ArrowArrayBuffer(self.array_.get(), 1);
  return data_buffer->size_bytes;
}

template <typename T> int64_t size(T &&self) { return self.array_->length; }

template <typename T> bool any(T &&self) {
  return self.array_->length > self.array_->null_count;
}

template <typename T> bool all(T &&self) {
  return self.array_->null_count == 0;
}

template <typename T> BoolArray isna(T &&self) {}

template <typename T> T take(T &&self, const std::vector<int64_t> &indices) {}

template <typename T> T copy(T &&self) {}

template <typename T> T fillna(T &&self) {}

template <typename T> T dropna(T &&self) {}

template <typename T> Int64Array len(T &&self) = delete;
