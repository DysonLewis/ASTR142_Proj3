/*
 * N-body gravitational acceleration calculation in C++
 * Adapted from Project 1's accel.c implementation
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>

#define G 6.67259e-8  // [cm^3/g/s^2]

static PyObject* get_accel([[maybe_unused]] PyObject* self, PyObject* args) {
    PyArrayObject* R_arr = nullptr;
    PyArrayObject* M_arr = nullptr;
    PyArrayObject* A_arr = nullptr;
    
    if (!PyArg_ParseTuple(args, "O!O!", 
                          &PyArray_Type, &R_arr,
                          &PyArray_Type, &M_arr)) {
        return nullptr;
    }
    
    if (PyArray_NDIM(R_arr) != 2 || PyArray_DIM(R_arr, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, "R must be N x 3 array");
        return nullptr;
    }
    
    if (PyArray_NDIM(M_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "M must be 1D array");
        return nullptr;
    }
    
    int N = PyArray_DIM(R_arr, 0);
    
    if (PyArray_DIM(M_arr, 0) != N) {
        PyErr_SetString(PyExc_ValueError, "R and M must have same number of particles");
        return nullptr;
    }
    
    npy_intp dims[2] = {N, 3};
    A_arr = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (!A_arr) {
        return nullptr;
    }
    
    double* R = (double*)PyArray_DATA(R_arr);
    double* M = (double*)PyArray_DATA(M_arr);
    double* A = (double*)PyArray_DATA(A_arr);
    
    for (int i = 0; i < N; i++) {
        double ax = 0.0, ay = 0.0, az = 0.0;
        
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            
            double dx = R[i*3 + 0] - R[j*3 + 0];
            double dy = R[i*3 + 1] - R[j*3 + 1];
            double dz = R[i*3 + 2] - R[j*3 + 2];
            
            double r2 = dx*dx + dy*dy + dz*dz;
            double ir3 = std::pow(r2, -1.5);
            
            double factor = G * M[j] * ir3;
            ax -= factor * dx;
            ay -= factor * dy;
            az -= factor * dz;
        }
        
        A[i*3 + 0] = ax;
        A[i*3 + 1] = ay;
        A[i*3 + 2] = az;
    }
    
    return (PyObject*)A_arr;
}

static PyMethodDef accel_methods[] = {
    {"get_accel", get_accel, METH_VARARGS, 
     "Compute gravitational acceleration for N bodies"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef accel_module = {
    PyModuleDef_HEAD_INIT,
    "accel",
    "N-body gravitational acceleration module",
    -1,
    accel_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyMODINIT_FUNC PyInit_accel(void) {
    import_array();
    return PyModule_Create(&accel_module);
}