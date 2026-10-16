import io
import os
import tempfile
import unittest

import numpy as np

import nrrd


class TestAsciiIntegerPrecision(unittest.TestCase):
    @staticmethod
    def integer_cases():
        return (
            ('int64', [-9223372036854775808, -9223372036854775807,
                       -9007199254740993, -1, 0, 9007199254740991,
                       9007199254740992, 9007199254740993,
                       9007199254740995, 9223372036854775807,
                       1234567890123456789, -1234567890123456789]),
            ('uint64', [0, 1, 9007199254740991, 9007199254740992,
                        9007199254740993, 9007199254740995,
                        9223372036854775807, 9223372036854775808,
                        9223372036854775809, 18446744073709551615,
                        18446744073709551614, 12345678901234567890]),
        )

    def assert_ascii_tokens(self, stream, data, order):
        # Independently inspect the wire representation, without using the reader.
        payload = stream.getvalue().split(b'\n\n', 1)[1]
        expected = [int(value) for value in data.ravel(order=order)]
        tokens = payload.split()
        self.assertEqual(tokens, [str(value).encode('ascii') for value in expected])
        self.assertEqual([int(token) for token in tokens], expected)

    def test_64bit_integer_tokens(self):
        for dtype, values in self.integer_cases():
            for shape in ((12,), (3, 4), (2, 2, 3)):
                for order in ('C', 'F'):
                    for encoding in ('ascii', 'ASCII', 'text', 'txt'):
                        with self.subTest(dtype=dtype, shape=shape, order=order, encoding=encoding):
                            data = np.array(values, dtype=dtype).reshape(shape)
                            stream = io.BytesIO()
                            nrrd.write(stream, data, {'encoding': encoding}, index_order=order)
                            self.assert_ascii_tokens(stream, data, order)

    def test_64bit_integer_roundtrip(self):
        for dtype, values in self.integer_cases():
            for shape in ((12,), (3, 4), (2, 2, 3)):
                for order in ('C', 'F'):
                    for storage in ('memory', 'attached', 'detached'):
                        with self.subTest(dtype=dtype, shape=shape, order=order, storage=storage):
                            data = np.array(values, dtype=dtype).reshape(shape)
                            with tempfile.TemporaryDirectory() as directory:
                                if storage == 'memory':
                                    stream = io.BytesIO()
                                    nrrd.write(stream, data, {'encoding': 'ascii'}, index_order=order)
                                    stream.seek(0)
                                    header = nrrd.read_header(stream)
                                    actual = nrrd.read_data(header, stream, index_order=order)
                                else:
                                    suffix = '.nhdr' if storage == 'detached' else '.nrrd'
                                    filename = os.path.join(directory, 'integers' + suffix)
                                    nrrd.write(filename, data, {'encoding': 'ascii'}, index_order=order)
                                    actual, header = nrrd.read(filename, index_order=order)
                            self.assertEqual(actual.dtype, data.dtype)
                            np.testing.assert_array_equal(actual, data)

    def test_small_integer_types(self):
        for dtype in ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32'):
            limits = np.iinfo(dtype)
            data = np.array([limits.min, 0, 1, limits.max], dtype=dtype)
            with self.subTest(dtype=dtype):
                stream = io.BytesIO()
                nrrd.write(stream, data, {'encoding': 'ascii'})
                self.assert_ascii_tokens(stream, data, 'F')
                stream.seek(0)
                header = nrrd.read_header(stream)
                np.testing.assert_array_equal(nrrd.read_data(header, stream), data)

    def test_noncontiguous_and_nonnative_integers(self):
        for dtype, values in self.integer_cases():
            for order in ('C', 'F'):
                # Reverse the byte order without changing the represented values.
                nonnative = np.dtype(dtype).newbyteorder('S')
                data = np.array(values, dtype=nonnative).reshape(3, 4).T[:, ::-1]
                with self.subTest(dtype=dtype, order=order):
                    self.assertFalse(data.flags.c_contiguous)
                    self.assertFalse(data.flags.f_contiguous)
                    stream = io.BytesIO()
                    nrrd.write(stream, data, {'encoding': 'ascii'}, index_order=order)
                    self.assert_ascii_tokens(stream, data, order)
                    stream.seek(0)
                    header = nrrd.read_header(stream)
                    np.testing.assert_array_equal(nrrd.read_data(header, stream, index_order=order), data)

    def test_floating_format_unchanged(self):
        for dtype in ('float32', 'float64'):
            limits = np.finfo(dtype)
            values = [0., -0., 0.1, -1.5, limits.max, limits.tiny,
                      np.nextafter(np.array(0, dtype=dtype), np.array(1, dtype=dtype)),
                      np.inf, -np.inf, np.nan, 1., -limits.max]
            for shape in ((12,), (3, 4), (2, 2, 3)):
                for order in ('C', 'F'):
                    with self.subTest(dtype=dtype, shape=shape, order=order):
                        data = np.array(values, dtype=dtype).reshape(shape)
                        stream = io.BytesIO()
                        nrrd.write(stream, data, {'encoding': 'ascii'}, index_order=order)
                        expected = io.BytesIO()
                        formatted = data.ravel(order=order) if data.ndim > 2 else data if order == 'C' else data.T
                        np.savetxt(expected, formatted, '%.17g')
                        self.assertEqual(stream.getvalue().split(b'\n\n', 1)[1], expected.getvalue())
                        stream.seek(0)
                        header = nrrd.read_header(stream)
                        actual = nrrd.read_data(header, stream, index_order=order)
                        np.testing.assert_array_equal(actual, data)
                        np.testing.assert_array_equal(np.signbit(actual), np.signbit(data))


if __name__ == '__main__':
    unittest.main()
