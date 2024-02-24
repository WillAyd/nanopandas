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


@pytest.fixture
def use_numpy():
    return False


class TestStringExtension(base.ExtensionTests):
    pass
