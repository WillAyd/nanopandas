#pragma once

#include "../array_types.hpp"

StringArray Lower(const StringArray &self);
StringArray Upper(const StringArray &self);
StringArray Capitalize(const StringArray &self);

template <typename T> Int64Array len(const T &self);
