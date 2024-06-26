import pytest

import nanopandas as nanopd


def test_from_sequence():
    # TODO: this is not really a classmethod
    # See https://github.com/wjakob/nanobind/discussions/416
    arr = nanopd.StringArray([])._from_sequence(("foo", None, "bar", "baz"))
    assert arr[0] == "foo"
    assert arr[1] is None
    assert arr[2] == "bar"
    assert arr[3] == "baz"


def test_from_factorized():
    # TODO: this is not really a classmethod
    # See https://github.com/wjakob/nanobind/discussions/416
    base_arr = nanopd.StringArray(["foo", "bar"])
    locs_arr = nanopd.Int64Array([0, -1, 1, 0, -1])
    result = nanopd.StringArray([])._from_factorized(locs_arr, base_arr)

    assert result.to_pylist() == ["foo", None, "bar", "foo", None]


def test_getitem():
    arr = nanopd.StringArray(["foo", None, "bar", "baz"])
    assert arr[0] == "foo"
    assert arr[1] is None
    assert arr[2] == "bar"
    assert arr[3] == "baz"


def test_getitem_slice():
    arr = nanopd.StringArray(["foo", None, "bar", "baz"])
    result = arr[:2]
    assert len(result) == 2
    assert result[0] == "foo"
    assert result[1] is None

def test_getitem_slice_neg_start():
    arr = nanopd.StringArray(["foo", None, "bar", "baz"])
    result = arr[-2:]
    assert len(result) == 2
    assert result[0] == "bar"
    assert result[1] == "baz"


def test_getitem_slice_neg_step():
    arr = nanopd.StringArray(["foo", None, "bar", "baz"])
    result = arr[2:0:-1]
    assert len(result) == 2
    assert result[0] == "bar"
    assert result[1] is None


def test_len():
    arr = nanopd.StringArray(["foo", "bar", "baz"])
    assert len(arr) == 3


def test_eq():
    arr = nanopd.StringArray(["foo", None, "foo", "üàéµ", "üàéµ"])
    other = nanopd.StringArray(["foo", None, "42", "üàéµ", "üàéµ"])
    result = arr == other
    assert result.to_pylist() == [True, None, False, True, True]


def test_nbytes():
    arr = nanopd.StringArray(["foo", "bar", "baz"])
    assert arr.nbytes == 32


def test_shape():
    arr = nanopd.StringArray(["foo", "bar", "baz"])
    assert arr.shape == (3,)


def test_isna():
    arr = nanopd.StringArray(["foo", None, "bar", None, "baz"])
    result = arr.isna()
    assert result.to_pylist() == [False, True, False, True, False]

def test_isna_no_missing():
    arr = nanopd.StringArray(["foo", "bar", "baz"])
    result = arr.isna()
    assert result.to_pylist() == [False, False, False]

def test_isna_no_data():
    arr = nanopd.StringArray([])
    result = arr.isna()
    assert result.to_pylist() == []


def test_take():
    arr = nanopd.StringArray(["foo", None, "bar", None, "baz"])
    result = arr.take([0, 1, 1, 0])
    assert result.to_pylist() == ["foo", None, None, "foo"]


def test_negative_take():
    arr = nanopd.StringArray(["foo", None, "bar", None, "baz"])
    result = arr.take([0, -1, -1, 0])
    assert result.to_pylist() == ["foo", "baz", "baz", "foo"]


def test_copy():
    arr = nanopd.StringArray(["foo", None, "bar", None, "baz"])
    result = arr.copy()
    assert arr.to_pylist() == result.to_pylist()


def test_concat_same_type():
    arr = nanopd.StringArray(["foo", None, "bar", None, "baz"])
    other = nanopd.StringArray([None, "quz", None, "quux"])
    result = arr._concat_same_type(other)

    expected = ["foo", None, "bar", None, "baz", None, "quz", None, "quux"]
    assert result.to_pylist() == expected


def test_interpolate():
    arr = nanopd.StringArray([None, "foo", None, "bar", None, "baz"])
    result = arr.interpolate()
    expected = [None, "foo", "foo", "bar", "bar", "baz"]
    assert result.to_pylist() == expected


def test_repr():
    arr = nanopd.StringArray([None, "foo", "bar", "baz"])
    result = repr(arr)
    expected = """StringArray
[null, "foo", "bar", "baz"]"""

    assert result == expected


@pytest.mark.skip("not important for now")
def test_repr_truncation():
    arr = nanopd.StringArray(["a" * 40])
    second_line = f'["{"a" * 38}'
    result = repr(arr)
    expected = f"""StringArray
{second_line}"""

    assert result == expected


def test_fillna():
    arr = nanopd.StringArray([None, "foo", None, "bar", None, "baz"])
    result = arr.fillna("filled")
    expected = ["filled", "foo", "filled", "bar", "filled", "baz"]
    assert result.to_pylist() == expected


def test_pad_or_backfill_pad():
    arr = nanopd.StringArray([None, "foo", None, None, "baz", None])
    result = arr._pad_or_backfill("pad")
    expected = [None, "foo", "foo", "foo", "baz", "baz"]
    assert result.to_pylist() == expected


def test_pad_or_backfill_backfill():
    arr = nanopd.StringArray([None, "foo", None, None, "baz", None])
    result = arr._pad_or_backfill("backfill")
    expected = ["foo", "foo", "baz", "baz", "baz", None]
    assert result.to_pylist() == expected


def test_dropna():
    arr = nanopd.StringArray([None, "foo", None, "bar", None, "baz"])
    result = arr.dropna()
    expected = ["foo", "bar", "baz"]
    assert result.to_pylist() == expected


def test_unique():
    arr = nanopd.StringArray(["foo", None, "foo", "üàéµ", "üàéµ"])
    result = arr.unique()

    expected = {"foo", "üàéµ"}
    assert len(expected) == len(result)
    for i in range(len(expected)):
        expected.remove(result[i])


def test_factorize():
    arr = nanopd.StringArray(["foo", None, "foo", "üàéµ", "üàéµ"])
    locs, uniqs = arr.factorize()

    assert locs.to_pylist() == [0, -1, 0, 1, 1]
    assert uniqs.to_pylist() == ["foo", "üàéµ"]


# str accessor methods
def test_len():
    arr = nanopd.StringArray(["foo", None, "bar", "üàéµ", "baz"])
    result = arr.len()
    assert result.to_pylist() == [3, None, 3, 4, 3]


def test_lower():
    arr = nanopd.StringArray(["FOO", None, "BAR", "ÜÀÉΜ", "baz"])
    result = arr.lower()
    assert result.to_pylist() == ["foo", None, "bar", "üàéμ", "baz"]


def test_upper():
    arr = nanopd.StringArray(["foo", None, "bar", "üàéµ", "BAZ"])
    result = arr.upper()
    assert result.to_pylist() == ["FOO", None, "BAR", "ÜÀÉΜ", "BAZ"]


def test_capitalize():
    arr = nanopd.StringArray(["foo", None, "bar", "üàéµ", "BAZ"])
    result = arr.capitalize()
    assert result.to_pylist() == ["Foo", None, "Bar", "Üàéµ", "BAZ"]


def test_isalnum():
    arr = nanopd.StringArray(["foo", None, "üàéµ", "bar!!", "42", " "])
    result = arr.isalnum()
    assert result.to_pylist() == [True, None, True, False, True, False]


def test_isalnum_trailing_nones():
    arr = nanopd.StringArray(["a", "bbbbb", "cc", None])
    result = arr.isalnum()
    assert result.to_pylist() == [True, True, True, None]


def test_isalpha():
    arr = nanopd.StringArray(["foo", None, "üàéµ", "bar!!", "42", " "])
    result = arr.isalpha()
    assert result.to_pylist() == [True, None, True, False, False, False]


def test_isdigit():
    arr = nanopd.StringArray(["foo", None, "üàéµ", "bar!!", "42", " "])
    result = arr.isdigit()
    assert result.to_pylist() == [False, None, False, False, True, False]


def test_isspace():
    arr = nanopd.StringArray(["foo", None, "üàéµ", "bar!!", "42", " "])
    result = arr.isspace()
    assert result.to_pylist() == [False, None, False, False, False, True]


def test_islower():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    result = arr.islower()
    assert result.to_pylist() == [False, None, True, False, True]


def test_isupper():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    result = arr.isupper()
    assert result.to_pylist() == [True, None, False, True, False]


def test_size():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    assert arr.size == 5


def test_null_count():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    assert arr.null_count == 1

def test_any():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    assert arr.any() == True


def test_all():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    assert arr.all() == False


# dtype tests
def test_dtype_str():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    assert str(arr.dtype) == "string[nanoarrow]"


def test_dtype_na_value():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    assert arr.dtype.na_value is None


def test_dtype_kind():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    assert arr.dtype.kind == "O"


def test_dtype_name():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    assert arr.dtype.name == "string[nanoarrow]"

def test_dtype_is_numeric():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    assert not arr.dtype.is_numeric


def test_dtype_is_boolean():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    assert not arr.dtype.is_boolean
