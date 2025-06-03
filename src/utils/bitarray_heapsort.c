#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <omp.h>
#include <string.h>

typedef struct {
    unsigned char* data;
    int row;
    int cols;
    npy_intp stride;  // Use npy_intp for stride to match numpy
} BitRow;

// Compare rows element by element, column by column, with 1 > 0
static int compare_bit_rows(const void* a, const void* b) {
    const BitRow* row_a = (const BitRow*)a;
    const BitRow* row_b = (const BitRow*)b;
    unsigned char* data_a = row_a->data + row_a->row * row_a->stride;
    unsigned char* data_b = row_b->data + row_b->row * row_b->stride;
    
    // Compare element by element from left to right
    for (int i = 0; i < row_a->cols; i++) {
        if (data_a[i] > data_b[i]) return -1; // 1 > 0, so row_a comes first
        if (data_a[i] < data_b[i]) return 1;  // 0 < 1, so row_b comes first
    }
    return 0; // Rows are identical
}

static PyObject* sort_numpy_array(PyObject* self, PyObject* args) {
    PyArrayObject* array = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array))
        return NULL;
    if (PyArray_NDIM(array) != 2 || PyArray_TYPE(array) != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "Input must be a 2D numpy array of dtype uint8");
        return NULL;
    }
    
    // Ensure array is C-contiguous
    PyArrayObject* c_array = (PyArrayObject*)PyArray_GETCONTIGUOUS(array);
    if (!c_array) return NULL;
    
    npy_intp* dims = PyArray_DIMS(c_array);
    int rows = (int)dims[0], cols = (int)dims[1];
    PyArrayObject* result = (PyArrayObject*)PyArray_NewLikeArray(c_array, NPY_CORDER, NULL, 0);
    if (!result) { 
        Py_DECREF(c_array); 
        return NULL; 
    }
    
    BitRow* bit_rows = (BitRow*)malloc(rows * sizeof(BitRow));
    if (!bit_rows) { 
        Py_DECREF(result); 
        Py_DECREF(c_array); 
        return NULL; 
    }
    
    unsigned char* data = (unsigned char*)PyArray_DATA(c_array);
    npy_intp stride = PyArray_STRIDES(c_array)[0];
    
    // Initialize BitRow structures
    for (int i = 0; i < rows; i++) {
        bit_rows[i].data = data;
        bit_rows[i].row = i;
        bit_rows[i].cols = cols;
        bit_rows[i].stride = stride;
    }
    
    // Sort rows using element-by-element comparison
    qsort(bit_rows, rows, sizeof(BitRow), compare_bit_rows);
    
    unsigned char* result_data = (unsigned char*)PyArray_DATA(result);
    npy_intp result_stride = PyArray_STRIDES(result)[0];
    
    // Copy sorted rows to result
    for (int i = 0; i < rows; i++) {
        unsigned char* src = data + bit_rows[i].row * stride;
        unsigned char* dst = result_data + i * result_stride;
        memcpy(dst, src, stride);  // Copy the entire row stride
    }
    
    free(bit_rows);
    Py_DECREF(c_array);
    return (PyObject*)result;
}

static PyMethodDef BitArraySortMethods[] = {
    {"sort_numpy", sort_numpy_array, METH_VARARGS, "Sort a 2D numpy array of bits (1 > 0)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef bitarray_heapsort_module = {
    PyModuleDef_HEAD_INIT,
    "bitarray_heapsort",
    "Module for sorting bit arrays with 1 > 0 ordering",
    -1,
    BitArraySortMethods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_bitarray_heapsort(void) {
    import_array();
    return PyModule_Create(&bitarray_heapsort_module);
}