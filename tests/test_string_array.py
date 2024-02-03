import nanopandas as nanopd


def test_getitem():
    arr = nanopd.StringArray(["foo", "bar", "baz"])
    assert arr[0] == "foo"
    assert arr[1] == "bar"
    assert arr[2] == "baz"

    
def test_len():
    arr = nanopd.StringArray(["foo", "bar", "baz"])
    assert len(arr) == 3


def test_dtype():
    arr = nanopd.StringArray(["foo", "bar", "baz"])
    assert arr.dtype == "string[arrow]"

def test_nbytes():
    arr = nanopd.StringArray(["foo", "bar", "baz"])
    assert arr.nbytes == 32

    
def test_isna():
    arr = nanopd.StringArray(["foo", None, "bar", None, "baz"])
    result = arr.isna()
    assert result.to_pylist() == [False, True, False, True, False]

    
def test_take():
    arr = nanopd.StringArray(["foo", None, "bar", None, "baz"])
    result = arr.take([0, 1, 1, 0])
    assert result.to_pylist() == ["foo", None, None, "foo"]


def test_unique():
    arr = nanopd.StringArray(["foo", None, "foo", "üàéµ", "üàéµ"])
    result = arr.unique()

    expected = {"foo", "üàéµ"}
    assert len(expected) == len(result)
    for i in range(len(expected)):
        expected.remove(result[i])


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


def test_any():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    assert arr.any() == True


def test_all():
    arr = nanopd.StringArray(["FOO", None, "foo", "ÜÀÉΜ", "üàéµ"])
    assert arr.all() == False
