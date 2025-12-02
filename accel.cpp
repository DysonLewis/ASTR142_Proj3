/*
 * N-body gravitational acceleration calculation in C++
 * Ultra-precision version using long double + Kahan summation
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>

#define G 6.67259e-8L  // [cm^3/g/s^2]

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
    
    double* R_in = (double*)PyArray_DATA(R_arr);
    double* M_in = (double*)PyArray_DATA(M_arr);
    double* A = (double*)PyArray_DATA(A_arr);
    
    for (int i = 0; i < N; i++) {
        // Use long double for all internal calculations
        long double ax = 0.0L, ay = 0.0L, az = 0.0L;
        long double cx = 0.0L, cy = 0.0L, cz = 0.0L;  // Kahan compensation
        
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            
            long double dx = (long double)R_in[i*3 + 0] - (long double)R_in[j*3 + 0];
            long double dy = (long double)R_in[i*3 + 1] - (long double)R_in[j*3 + 1];
            long double dz = (long double)R_in[i*3 + 2] - (long double)R_in[j*3 + 2];
            
            long double r2 = dx*dx + dy*dy + dz*dz;
            long double r = sqrtl(r2);
            long double r3 = r2 * r;
            long double ir3 = 1.0L / r3;
            
            long double factor = G * (long double)M_in[j] * ir3;
            
            // Kahan summation with long double
            long double term_x = -factor * dx - cx;
            long double temp_x = ax + term_x;
            cx = (temp_x - ax) - term_x;
            ax = temp_x;
            
            long double term_y = -factor * dy - cy;
            long double temp_y = ay + term_y;
            cy = (temp_y - ay) - term_y;
            ay = temp_y;
            
            long double term_z = -factor * dz - cz;
            long double temp_z = az + term_z;
            cz = (temp_z - az) - term_z;
            az = temp_z;
        }
        
        A[i*3 + 0] = (double)ax;
        A[i*3 + 1] = (double)ay;
        A[i*3 + 2] = (double)az;
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
    "N-body gravitational acceleration module (ultra precision)",
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