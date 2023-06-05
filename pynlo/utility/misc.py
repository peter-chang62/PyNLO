# -*- coding: utf-8 -*-
"""
Miscellaneous helper classes and functions.

"""

__all__ = ["replace"]


# %% Imports

import numpy as np

# %% Helper Functions

def replace(array, values, key):
    """Copy `array` with elements given by `key` replaced by `values`."""
    array = array.copy()
    array[key] = values
    return array

# %% Array Properties for Classes

class ArrayWrapper(np.lib.mixins.NDArrayOperatorsMixin):
    """Emulates an array using custom item getters and setters."""
    def __init__(self, getter=None, setter=None):
        self._getter = getter
        self._setter = setter

    def __getitem__(self, key):
        return self._getter(key)

    def __setitem__(self, key, value):
        self._setter(key, value)

    def __array__(self, dtype=None):
        array = self.__getitem__(...)
        if dtype is None:
            return array
        else:
            return array.astype(dtype=dtype)

    def __repr__(self):
        return repr(self.__array__())

    def __len__(self):
        return len(self.__array__())

    def __copy__(self):
        return self.__array__()

    def __deepcopy__(self, memo):
        return self.__array__().copy()

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """
        Implemented to support use of the `out` ufunc keyword.

        Modified from NumPy docs, "__array_ufunc__ for ufuncs"

        """
        #---- Convert Input to Arrays
        inputs = tuple(x.__array__() if isinstance(x, ArrayWrapper) else x
                       for x in inputs)

        #---- Apply Ufunc
        if out:
            # Convert Output to Arrays
            outputs = []
            out_args = []
            for idx, output in enumerate(out):
                if isinstance(output, ArrayWrapper):
                    outputs.append([idx, output])
                    out_args.append(output.__array__())
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)

            # Apply Ufunc
            result = getattr(ufunc, method)(*inputs, **kwargs)

            # Convert Output to ArrayWrappers
            for idx, output in outputs:
                output[...] = out_args[idx] # "in place" equivalent
        else:
            result = getattr(ufunc, method)(*inputs, **kwargs)

        #---- Return Result
        if method == 'at':
            return None # no return value
        else:
            return result

    def __getattr__(self, attr):
        """Catch-all for other numpy functions"""
        return getattr(self.__array__(), attr)


class SettableArrayProperty(property):
    """
    A subclass of `property` that allows extending the getter and setter
    formalism to Numpy array elements.

    Notes
    -----
    To allow usage of both `__get__`/`__getitem__` and `__set__`/`__setitem__`,
    the methods fed into `SettableArrayProperty` must contain a keyword
    argument and logic for processing the keys used by `__getitem__` and
    `__setitem__`. In the `setter` method, the `value` parameter must precede
    the `key` parameter. In the following example, the default key is an open
    slice (ellipsis), the entire array is retrieved when individual elements
    are not requested.::

        class C(object):
            def __init__(self):
                self.x = np.array([1,2,3,4])

            @SettableArrayProperty
            def y(self, key=...):
                return self.x[key]**2

            @y.setter
            def y(self, value, key=...):
                self.x[key] = value**0.5

    See the documentation of `property` for other implementation details.

    """
    def __get__(self, obj, objtype):
        # Return self if not instantiated
        if obj is None:
            return self

        # Define Item Getter and Setter
        def item_getter(key):
            return self.fget(obj, key)

        def item_setter(key, value):
            if self.fset is None:
                self.__set__(obj, value) # raise AttributeError if fset is None
            self.fset(obj, value, key)

        # Return array with custom item getters and setters
        array = ArrayWrapper(getter=item_getter, setter=item_setter)
        return array
