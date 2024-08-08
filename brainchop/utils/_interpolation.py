# :: from scipy.ndimage._ni_docstrings.py
"""Docstring components common to several ndimage functions."""
import warnings

import numpy as np

from scipy.ndimage import _nd_image
from scipy._lib._util import normalize_axis_index
from scipy._lib import doccer

import _ni_support

__all__ = ['docfiller', 'spline_filter1d', 'spline_filter', 'geometric_transform',
           'affine_transform']


_input_doc = (
"""input : array_like
    The input array.""")
_axis_doc = (
"""axis : int, optional
    The axis of `input` along which to calculate. Default is -1.""")
_output_doc = (
"""output : array or dtype, optional
    The array in which to place the output, or the dtype of the
    returned array. By default an array of the same dtype as input
    will be created.""")
_size_foot_doc = (
"""size : scalar or tuple, optional
    See footprint, below. Ignored if footprint is given.
footprint : array, optional
    Either `size` or `footprint` must be defined. `size` gives
    the shape that is taken from the input array, at every element
    position, to define the input to the filter function.
    `footprint` is a boolean array that specifies (implicitly) a
    shape, but also which of the elements within this shape will get
    passed to the filter function. Thus ``size=(n,m)`` is equivalent
    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
    of dimensions of the input array, so that, if the input array is
    shape (10,10,10), and `size` is 2, then the actual size used is
    (2,2,2). When `footprint` is given, `size` is ignored.""")
_mode_reflect_doc = (
"""mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'reflect'. Behavior for each valid
    value is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel. This mode is also sometimes referred to as half-sample
        symmetric.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel. This mode is also sometimes referred to as whole-sample
        symmetric.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.

    For consistency with the interpolation functions, the following mode
    names can also be used:

    'grid-mirror'
        This is a synonym for 'reflect'.

    'grid-constant'
        This is a synonym for 'constant'.

    'grid-wrap'
        This is a synonym for 'wrap'.""")

_mode_interp_constant_doc = (
"""mode : {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', \
'mirror', 'grid-wrap', 'wrap'}, optional
    The `mode` parameter determines how the input array is extended
    beyond its boundaries. Default is 'constant'. Behavior for each valid
    value is as follows (see additional plots and details on
    :ref:`boundary modes <ndimage-interpolation-modes>`):

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel. This mode is also sometimes referred to as half-sample
        symmetric.

    'grid-mirror'
        This is a synonym for 'reflect'.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter. No
        interpolation is performed beyond the edges of the input.

    'grid-constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter. Interpolation
        occurs for samples outside the input's extent  as well.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel. This mode is also sometimes referred to as whole-sample
        symmetric.

    'grid-wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.

    'wrap' (`d b c d | a b c d | b c a b`)
        The input is extended by wrapping around to the opposite edge, but in a
        way such that the last point and initial point exactly overlap. In this
        case it is not well defined which sample will be chosen at the point of
        overlap.""")
_mode_interp_mirror_doc = (
    _mode_interp_constant_doc.replace("Default is 'constant'",
                                      "Default is 'mirror'")
)
assert _mode_interp_mirror_doc != _mode_interp_constant_doc, \
    'Default not replaced'

_mode_multiple_doc = (
"""mode : str or sequence, optional
    The `mode` parameter determines how the input array is extended
    when the filter overlaps a border. By passing a sequence of modes
    with length equal to the number of dimensions of the input array,
    different modes can be specified along each axis. Default value is
    'reflect'. The valid values and their behavior is as follows:

    'reflect' (`d c b a | a b c d | d c b a`)
        The input is extended by reflecting about the edge of the last
        pixel. This mode is also sometimes referred to as half-sample
        symmetric.

    'constant' (`k k k k | a b c d | k k k k`)
        The input is extended by filling all values beyond the edge with
        the same constant value, defined by the `cval` parameter.

    'nearest' (`a a a a | a b c d | d d d d`)
        The input is extended by replicating the last pixel.

    'mirror' (`d c b | a b c d | c b a`)
        The input is extended by reflecting about the center of the last
        pixel. This mode is also sometimes referred to as whole-sample
        symmetric.

    'wrap' (`a b c d | a b c d | a b c d`)
        The input is extended by wrapping around to the opposite edge.

    For consistency with the interpolation functions, the following mode
    names can also be used:

    'grid-constant'
        This is a synonym for 'constant'.

    'grid-mirror'
        This is a synonym for 'reflect'.

    'grid-wrap'
        This is a synonym for 'wrap'.""")
_cval_doc = (
"""cval : scalar, optional
    Value to fill past edges of input if `mode` is 'constant'. Default
    is 0.0.""")
_origin_doc = (
"""origin : int, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right.""")
_origin_multiple_doc = (
"""origin : int or sequence, optional
    Controls the placement of the filter on the input array's pixels.
    A value of 0 (the default) centers the filter over the pixel, with
    positive values shifting the filter to the left, and negative ones
    to the right. By passing a sequence of origins with length equal to
    the number of dimensions of the input array, different shifts can
    be specified along each axis.""")
_extra_arguments_doc = (
"""extra_arguments : sequence, optional
    Sequence of extra positional arguments to pass to passed function.""")
_extra_keywords_doc = (
"""extra_keywords : dict, optional
    dict of extra keyword arguments to pass to passed function.""")
_prefilter_doc = (
"""prefilter : bool, optional
    Determines if the input array is prefiltered with `spline_filter`
    before interpolation. The default is True, which will create a
    temporary `float64` array of filtered values if ``order > 1``. If
    setting this to False, the output will be slightly blurred if
    ``order > 1``, unless the input is prefiltered, i.e. it is the result
    of calling `spline_filter` on the original input.""")

docdict = {
    'input': _input_doc,
    'axis': _axis_doc,
    'output': _output_doc,
    'size_foot': _size_foot_doc,
    'mode_interp_constant': _mode_interp_constant_doc,
    'mode_interp_mirror': _mode_interp_mirror_doc,
    'mode_reflect': _mode_reflect_doc,
    'mode_multiple': _mode_multiple_doc,
    'cval': _cval_doc,
    'origin': _origin_doc,
    'origin_multiple': _origin_multiple_doc,
    'extra_arguments': _extra_arguments_doc,
    'extra_keywords': _extra_keywords_doc,
    'prefilter': _prefilter_doc
    }

docfiller = doccer.filldoc(docdict)

# :: from scipy.ndimage._interpolation
# Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@docfiller
def spline_filter1d(input, order=3, axis=-1, output=np.float64,
                    mode='mirror'):
    """
    Calculate a 1-D spline filter along the given axis.

    The lines of the array along the given axis are filtered by a
    spline filter. The order of the spline must be >= 2 and <= 5.

    Parameters
    ----------
    %(input)s
    order : int, optional
        The order of the spline, default is 3.
    axis : int, optional
        The axis along which the spline filter is applied. Default is the last
        axis.
    output : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array. Default is ``numpy.float64``.
    %(mode_interp_mirror)s

    Returns
    -------
    spline_filter1d : ndarray
        The filtered input.

    See Also
    --------
    spline_filter : Multidimensional spline filter.

    Notes
    -----
    All of the interpolation functions in `ndimage` do spline interpolation of
    the input image. If using B-splines of `order > 1`, the input image
    values have to be converted to B-spline coefficients first, which is
    done by applying this 1-D filter sequentially along all
    axes of the input. All functions that require B-spline coefficients
    will automatically filter their inputs, a behavior controllable with
    the `prefilter` keyword argument. For functions that accept a `mode`
    parameter, the result will only be correct if it matches the `mode`
    used when filtering.

    For complex-valued `input`, this function processes the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    We can filter an image using 1-D spline along the given axis:

    >>> from scipy.ndimage import spline_filter1d
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> orig_img = np.eye(20)  # create an image
    >>> orig_img[10, :] = 1.0
    >>> sp_filter_axis_0 = spline_filter1d(orig_img, axis=0)
    >>> sp_filter_axis_1 = spline_filter1d(orig_img, axis=1)
    >>> f, ax = plt.subplots(1, 3, sharex=True)
    >>> for ind, data in enumerate([[orig_img, "original image"],
    ...             [sp_filter_axis_0, "spline filter (axis=0)"],
    ...             [sp_filter_axis_1, "spline filter (axis=1)"]]):
    ...     ax[ind].imshow(data[0], cmap='gray_r')
    ...     ax[ind].set_title(data[1])
    >>> plt.tight_layout()
    >>> plt.show()

    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    complex_output = np.iscomplexobj(input)
    output = _ni_support._get_output(output, input,
                                     complex_output=complex_output)
    if complex_output:
        spline_filter1d(input.real, order, axis, output.real, mode)
        spline_filter1d(input.imag, order, axis, output.imag, mode)
        return output
    if order in [0, 1]:
        output[...] = np.array(input)
    else:
        mode = _ni_support._extend_mode_to_code(mode)
        axis = normalize_axis_index(axis, input.ndim)
        _nd_image.spline_filter1d(input, order, axis, output, mode)
    return output

@docfiller
def spline_filter(input, order=3, output=np.float64, mode='mirror'):
    """
    Multidimensional spline filter.

    Parameters
    ----------
    %(input)s
    order : int, optional
        The order of the spline, default is 3.
    output : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array. Default is ``numpy.float64``.
    %(mode_interp_mirror)s

    Returns
    -------
    spline_filter : ndarray
        Filtered array. Has the same shape as `input`.

    See Also
    --------
    spline_filter1d : Calculate a 1-D spline filter along the given axis.

    Notes
    -----
    The multidimensional filter is implemented as a sequence of
    1-D spline filters. The intermediate arrays are stored
    in the same data type as the output. Therefore, for output types
    with a limited precision, the results may be imprecise because
    intermediate results may be stored with insufficient precision.

    For complex-valued `input`, this function processes the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    We can filter an image using multidimentional splines:

    >>> from scipy.ndimage import spline_filter
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> orig_img = np.eye(20)  # create an image
    >>> orig_img[10, :] = 1.0
    >>> sp_filter = spline_filter(orig_img, order=3)
    >>> f, ax = plt.subplots(1, 2, sharex=True)
    >>> for ind, data in enumerate([[orig_img, "original image"],
    ...                             [sp_filter, "spline filter"]]):
    ...     ax[ind].imshow(data[0], cmap='gray_r')
    ...     ax[ind].set_title(data[1])
    >>> plt.tight_layout()
    >>> plt.show()

    """
    if order < 2 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    complex_output = np.iscomplexobj(input)
    output = _ni_support._get_output(output, input,
                                     complex_output=complex_output)
    if complex_output:
        spline_filter(input.real, order, output.real, mode)
        spline_filter(input.imag, order, output.imag, mode)
        return output
    if order not in [0, 1] and input.ndim > 0:
        for axis in range(input.ndim):
            spline_filter1d(input, order, axis, output=output, mode=mode)
            input = output
    else:
        output[...] = input[...]
    return output


def _prepad_for_spline_filter(input, mode, cval):
    if mode in ['nearest', 'grid-constant']:
        npad = 12
        if mode == 'grid-constant':
            padded = np.pad(input, npad, mode='constant',
                               constant_values=cval)
        elif mode == 'nearest':
            padded = np.pad(input, npad, mode='edge')
    else:
        # other modes have exact boundary conditions implemented so
        # no prepadding is needed
        npad = 0
        padded = input
    return padded, npad


@docfiller
def geometric_transform(input, mapping, output_shape=None,
                        output=None, order=3,
                        mode='constant', cval=0.0, prefilter=True,
                        extra_arguments=(), extra_keywords=None):
    """
    Apply an arbitrary geometric transform.

    The given mapping function is used to find, for each point in the
    output, the corresponding coordinates in the input. The value of the
    input at those coordinates is determined by spline interpolation of
    the requested order.

    Parameters
    ----------
    %(input)s
    mapping : {callable, scipy.LowLevelCallable}
        A callable object that accepts a tuple of length equal to the output
        array rank, and returns the corresponding input coordinates as a tuple
        of length equal to the input array rank.
    output_shape : tuple of ints, optional
        Shape tuple.
    %(output)s
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    %(mode_interp_constant)s
    %(cval)s
    %(prefilter)s
    extra_arguments : tuple, optional
        Extra arguments passed to `mapping`.
    extra_keywords : dict, optional
        Extra keywords passed to `mapping`.

    Returns
    -------
    output : ndarray
        The filtered input.

    See Also
    --------
    map_coordinates, affine_transform, spline_filter1d


    Notes
    -----
    This function also accepts low-level callback functions with one
    the following signatures and wrapped in `scipy.LowLevelCallable`:

    .. code:: c

       int mapping(npy_intp *output_coordinates, double *input_coordinates,
                   int output_rank, int input_rank, void *user_data)
       int mapping(intptr_t *output_coordinates, double *input_coordinates,
                   int output_rank, int input_rank, void *user_data)

    The calling function iterates over the elements of the output array,
    calling the callback function at each element. The coordinates of the
    current output element are passed through ``output_coordinates``. The
    callback function must return the coordinates at which the input must
    be interpolated in ``input_coordinates``. The rank of the input and
    output arrays are given by ``input_rank`` and ``output_rank``
    respectively. ``user_data`` is the data pointer provided
    to `scipy.LowLevelCallable` as-is.

    The callback function must return an integer error status that is zero
    if something went wrong and one otherwise. If an error occurs, you should
    normally set the Python error status with an informative message
    before returning, otherwise a default error message is set by the
    calling function.

    In addition, some other low-level function pointer specifications
    are accepted, but these are for backward compatibility only and should
    not be used in new code.

    For complex-valued `input`, this function transforms the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.ndimage import geometric_transform
    >>> a = np.arange(12.).reshape((4, 3))
    >>> def shift_func(output_coords):
    ...     return (output_coords[0] - 0.5, output_coords[1] - 0.5)
    ...
    >>> geometric_transform(a, shift_func)
    array([[ 0.   ,  0.   ,  0.   ],
           [ 0.   ,  1.362,  2.738],
           [ 0.   ,  4.812,  6.187],
           [ 0.   ,  8.263,  9.637]])

    >>> b = [1, 2, 3, 4, 5]
    >>> def shift_func(output_coords):
    ...     return (output_coords[0] - 3,)
    ...
    >>> geometric_transform(b, shift_func, mode='constant')
    array([0, 0, 0, 1, 2])
    >>> geometric_transform(b, shift_func, mode='nearest')
    array([1, 1, 1, 1, 2])
    >>> geometric_transform(b, shift_func, mode='reflect')
    array([3, 2, 1, 1, 2])
    >>> geometric_transform(b, shift_func, mode='wrap')
    array([2, 3, 4, 1, 2])

    """
    if extra_keywords is None:
        extra_keywords = {}
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    if output_shape is None:
        output_shape = input.shape
    if input.ndim < 1 or len(output_shape) < 1:
        raise RuntimeError('input and output rank must be > 0')
    complex_output = np.iscomplexobj(input)
    output = _ni_support._get_output(output, input, shape=output_shape,
                                     complex_output=complex_output)
    if complex_output:
        kwargs = dict(order=order, mode=mode, prefilter=prefilter,
                      output_shape=output_shape,
                      extra_arguments=extra_arguments,
                      extra_keywords=extra_keywords)
        geometric_transform(input.real, mapping, output=output.real,
                            cval=np.real(cval), **kwargs)
        geometric_transform(input.imag, mapping, output=output.imag,
                            cval=np.imag(cval), **kwargs)
        return output

    if prefilter and order > 1:
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        filtered = spline_filter(padded, order, output=np.float64,
                                 mode=mode)
    else:
        npad = 0
        filtered = input
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.geometric_transform(filtered, mapping, None, None, None, output,
                                  order, mode, cval, npad, extra_arguments,
                                  extra_keywords)
    return output


@docfiller
def affine_transform(input, matrix, offset=0.0, output_shape=None,
                     output=None, order=3,
                     mode='constant', cval=0.0, prefilter=True):
    """
    Apply an affine transformation.

    Given an output image pixel index vector ``o``, the pixel value
    is determined from the input image at position
    ``np.dot(matrix, o) + offset``.

    This does 'pull' (or 'backward') resampling, transforming the output space
    to the input to locate data. Affine transformations are often described in
    the 'push' (or 'forward') direction, transforming input to output. If you
    have a matrix for the 'push' transformation, use its inverse
    (:func:`numpy.linalg.inv`) in this function.

    Parameters
    ----------
    %(input)s
    matrix : ndarray
        The inverse coordinate transformation matrix, mapping output
        coordinates to input coordinates. If ``ndim`` is the number of
        dimensions of ``input``, the given matrix must have one of the
        following shapes:

            - ``(ndim, ndim)``: the linear transformation matrix for each
              output coordinate.
            - ``(ndim,)``: assume that the 2-D transformation matrix is
              diagonal, with the diagonal specified by the given value. A more
              efficient algorithm is then used that exploits the separability
              of the problem.
            - ``(ndim + 1, ndim + 1)``: assume that the transformation is
              specified using homogeneous coordinates [1]_. In this case, any
              value passed to ``offset`` is ignored.
            - ``(ndim, ndim + 1)``: as above, but the bottom row of a
              homogeneous transformation matrix is always ``[0, 0, ..., 1]``,
              and may be omitted.

    offset : float or sequence, optional
        The offset into the array where the transform is applied. If a float,
        `offset` is the same for each axis. If a sequence, `offset` should
        contain one value for each axis.
    output_shape : tuple of ints, optional
        Shape tuple.
    %(output)s
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    %(mode_interp_constant)s
    %(cval)s
    %(prefilter)s

    Returns
    -------
    affine_transform : ndarray
        The transformed input.

    Notes
    -----
    The given matrix and offset are used to find for each point in the
    output the corresponding coordinates in the input by an affine
    transformation. The value of the input at those coordinates is
    determined by spline interpolation of the requested order. Points
    outside the boundaries of the input are filled according to the given
    mode.

    .. versionchanged:: 0.18.0
        Previously, the exact interpretation of the affine transformation
        depended on whether the matrix was supplied as a 1-D or a
        2-D array. If a 1-D array was supplied
        to the matrix parameter, the output pixel value at index ``o``
        was determined from the input image at position
        ``matrix * (o + offset)``.

    For complex-valued `input`, this function transforms the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Homogeneous_coordinates
    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    if output_shape is None:
        if isinstance(output, np.ndarray):
            output_shape = output.shape
        else:
            output_shape = input.shape
    if input.ndim < 1 or len(output_shape) < 1:
        raise RuntimeError('input and output rank must be > 0')
    complex_output = np.iscomplexobj(input)
    output = _ni_support._get_output(output, input, shape=output_shape,
                                     complex_output=complex_output)
    if complex_output:
        kwargs = dict(offset=offset, output_shape=output_shape, order=order,
                      mode=mode, prefilter=prefilter)
        affine_transform(input.real, matrix, output=output.real,
                         cval=np.real(cval), **kwargs)
        affine_transform(input.imag, matrix, output=output.imag,
                         cval=np.imag(cval), **kwargs)
        return output
    if prefilter and order > 1:
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        filtered = spline_filter(padded, order, output=np.float64, mode=mode)
    else:
        npad = 0
        filtered = input
    mode = _ni_support._extend_mode_to_code(mode)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim not in [1, 2] or matrix.shape[0] < 1:
        raise RuntimeError('no proper affine matrix provided')
    if (matrix.ndim == 2 and matrix.shape[1] == input.ndim + 1 and
            (matrix.shape[0] in [input.ndim, input.ndim + 1])):
        if matrix.shape[0] == input.ndim + 1:
            exptd = [0] * input.ndim + [1]
            if not np.all(matrix[input.ndim] == exptd):
                msg = (f'Expected homogeneous transformation matrix with '
                       f'shape {matrix.shape} for image shape {input.shape}, '
                       f'but bottom row was not equal to {exptd}')
                raise ValueError(msg)
        # assume input is homogeneous coordinate transformation matrix
        offset = matrix[:input.ndim, input.ndim]
        matrix = matrix[:input.ndim, :input.ndim]
    if matrix.shape[0] != input.ndim:
        raise RuntimeError('affine matrix has wrong number of rows')
    if matrix.ndim == 2 and matrix.shape[1] != output.ndim:
        raise RuntimeError('affine matrix has wrong number of columns')
    if not matrix.flags.contiguous:
        matrix = matrix.copy()
    offset = _ni_support._normalize_sequence(offset, input.ndim)
    offset = np.asarray(offset, dtype=np.float64)
    if offset.ndim != 1 or offset.shape[0] < 1:
        raise RuntimeError('no proper offset provided')
    if not offset.flags.contiguous:
        offset = offset.copy()
    if matrix.ndim == 1:
        warnings.warn(
            "The behavior of affine_transform with a 1-D "
            "array supplied for the matrix parameter has changed in "
            "SciPy 0.18.0.",
            stacklevel=2
        )
        _nd_image.zoom_shift(filtered, matrix, offset/matrix, output, order,
                             mode, cval, npad, False)
    else:
        _nd_image.geometric_transform(filtered, None, None, matrix, offset,
                                      output, order, mode, cval, npad, None,
                                      None)
    return output
