import operator

import pytest
from pandas.api.extensions import ExtensionDtype
from pandas.tests.extension import base

import nanopandas as nanopd


# Could not actually inherit from ExtensionDtype - probably a nanobind limitation
# RuntimeError: nb_type_init(): invalid number of bases!


class NanoStringDtype(ExtensionDtype):

    @property
    def na_value(self):
        return None

    def type(self):
        return str

    def name(self):
        return "string[nanoarrow]"

    def __str__(self):
        return "NanoStringDtype"

    @classmethod
    def construct_array_type(cls):
        return nanopd.StringArray

    @property
    def _is_numeric(self):
        return False

    @property
    def _is_boolean(self):
        return False

    @property
    def _is_immutable(self) -> bool:
        return True


@pytest.fixture
def dtype():
    return NanoStringDtype()


@pytest.fixture
def data():
    return nanopd.StringArray(["foo", "bar"] * 50)


@pytest.fixture
def data_for_twos():
    pytest.skip("Not applicable for StringArray")


@pytest.fixture
def data_missing():
    return nanopd.StringArray([None, "foo"])


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_repeated(data):
    def gen(count):
        for _ in range(count):
            yield data

    return gen


@pytest.fixture
def data_for_sorting():
    return nanopd.StringArray(["baz", "foo", "bar"])


@pytest.fixture
def data_missing_for_sorting():
    return nanopd.StringArray(["foo", None, "bar"])


@pytest.fixture
def na_cmp():
    return operator.is_


@pytest.fixture
def na_value(dtype):
    return dtype.na_value


@pytest.fixture
def data_for_grouping():
    return nanopd.StringArray(["baz", "baz", None, None, "bar", "bar", "foo"])


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param

@pytest.fixture(params=[True, False])
def as_frame(request):
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request):
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
    return request.param


@pytest.fixture
def invalid_scalar(data):
    """
    A scalar that *cannot* be held by this ExtensionArray.

    The default should work for most subclasses, but is not guaranteed.

    If the array can hold any item (i.e. object dtype), then use pytest.skip.
    """
    return 42

@pytest.fixture
def use_numpy():
    return False


class TestStringExtension(base.ExtensionTests):
    pass
