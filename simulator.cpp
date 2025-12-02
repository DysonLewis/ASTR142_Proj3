/*
 * N-body simulation integration module in C++
 * Handles the main simulation loop with leapfrog integration
 * Uses accel module for acceleration calculations
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <vector>

#define G 6.6743e-8L  // [cm^3/g/s^2]

// Structure to hold simulation results
struct SimulationRecord {
    int simulation;
    double time_yr;
    int body_idx;
    double x, y, z;
    double vx, vy, vz;
    double KE, PE, E_tot;
};

// Call the accel module's get_accel function
static PyArrayObject* call_get_accel(PyArrayObject* X_arr, PyArrayObject* M_arr) {
    PyObject* accel_module = PyImport_ImportModule("accel");
    if (!accel_module) {
        PyErr_SetString(PyExc_ImportError, "Failed to import accel module");
        return nullptr;
    }
    
    PyObject* get_accel_func = PyObject_GetAttrString(accel_module, "get_accel");
    Py_DECREF(accel_module);
    
    if (!get_accel_func) {
        PyErr_SetString(PyExc_AttributeError, "Failed to find get_accel function");
        return nullptr;
    }
    
    PyObject* args = PyTuple_Pack(2, X_arr, M_arr);
    PyObject* result = PyObject_CallObject(get_accel_func, args);
    
    Py_DECREF(args);
    Py_DECREF(get_accel_func);
    
    if (!result) {
        return nullptr;
    }
    
    return (PyArrayObject*)result;
}

// Run a single simulation
static PyObject* run_simulation([[maybe_unused]] PyObject* self, PyObject* args) {
    PyArrayObject* X0_arr = nullptr;
    PyArrayObject* V0_arr = nullptr;
    PyArrayObject* M_arr = nullptr;
    PyArrayObject* perturb_idx_arr = nullptr;
    PyArrayObject* perturb_pos_arr = nullptr;
    PyArrayObject* perturb_vel_arr = nullptr;
    
    int sim_id, n_step;
    double dt, yr;
    
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!iidd",
                          &PyArray_Type, &X0_arr,
                          &PyArray_Type, &V0_arr,
                          &PyArray_Type, &M_arr,
                          &PyArray_Type, &perturb_idx_arr,
                          &PyArray_Type, &perturb_pos_arr,
                          &PyArray_Type, &perturb_vel_arr,
                          &sim_id,
                          &n_step,
                          &dt,
                          &yr)) {
        return nullptr;
    }
    
    int N = PyArray_DIM(X0_arr, 0);
    int n_perturb = PyArray_DIM(perturb_idx_arr, 0);
    
    // Copy initial conditions
    std::vector<double> X(N * 3);
    std::vector<double> V(N * 3);
    std::vector<double> M(N);
    
    double* X0_data = (double*)PyArray_DATA(X0_arr);
    double* V0_data = (double*)PyArray_DATA(V0_arr);
    double* M_data = (double*)PyArray_DATA(M_arr);
    
    for (int i = 0; i < N * 3; i++) {
        X[i] = X0_data[i];
        V[i] = V0_data[i];
    }
    for (int i = 0; i < N; i++) {
        M[i] = M_data[i];
    }
    
    // Apply perturbations
    int* perturb_idx = (int*)PyArray_DATA(perturb_idx_arr);
    double* perturb_pos = (double*)PyArray_DATA(perturb_pos_arr);
    double* perturb_vel = (double*)PyArray_DATA(perturb_vel_arr);
    
    for (int p = 0; p < n_perturb; p++) {
        int idx = perturb_idx[p];
        X[idx*3 + 0] += perturb_pos[p*3 + 0];
        X[idx*3 + 1] += perturb_pos[p*3 + 1];
        X[idx*3 + 2] += perturb_pos[p*3 + 2];
        V[idx*3 + 0] += perturb_vel[p*3 + 0];
        V[idx*3 + 1] += perturb_vel[p*3 + 1];
        V[idx*3 + 2] += perturb_vel[p*3 + 2];
    }
    
    // Prepare output arrays
    npy_intp dims[2] = {(npy_intp)(n_step * N), 11};
    PyArrayObject* results = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (!results) {
        return nullptr;
    }
    double* results_data = (double*)PyArray_DATA(results);
    
    // Create numpy arrays for positions and masses (for calling accel module)
    npy_intp pos_dims[2] = {N, 3};
    npy_intp mass_dims[1] = {N};
    
    PyArrayObject* X_arr = (PyArrayObject*)PyArray_SimpleNew(2, pos_dims, NPY_DOUBLE);
    PyArrayObject* M_arr_np = (PyArrayObject*)PyArray_SimpleNew(1, mass_dims, NPY_DOUBLE);
    
    if (!X_arr || !M_arr_np) {
        Py_XDECREF(X_arr);
        Py_XDECREF(M_arr_np);
        Py_DECREF(results);
        return nullptr;
    }
    
    // Copy masses to numpy array (only need to do this once)
    double* M_arr_data = (double*)PyArray_DATA(M_arr_np);
    for (int i = 0; i < N; i++) {
        M_arr_data[i] = M[i];
    }
    
    // Initial acceleration
    double* X_arr_data = (double*)PyArray_DATA(X_arr);
    for (int i = 0; i < N * 3; i++) {
        X_arr_data[i] = X[i];
    }
    
    PyArrayObject* A_arr = call_get_accel(X_arr, M_arr_np);
    if (!A_arr) {
        Py_DECREF(X_arr);
        Py_DECREF(M_arr_np);
        Py_DECREF(results);
        return nullptr;
    }
    
    double* A = (double*)PyArray_DATA(A_arr);
    
    // Integration loop (leapfrog)
    for (int step = 0; step < n_step; step++) {
        double t = step * dt;
        
        // V(t+dt/2) = V(t) + A(t) * dt/2
        for (int i = 0; i < N * 3; i++) {
            V[i] += A[i] * dt / 2.0;
        }
        
        // X(t+dt) = X(t) + V(t+dt/2) * dt
        for (int i = 0; i < N * 3; i++) {
            X[i] += V[i] * dt;
        }
        
        // A(t+dt) - update positions in numpy array and call accel
        for (int i = 0; i < N * 3; i++) {
            X_arr_data[i] = X[i];
        }
        
        Py_DECREF(A_arr);
        A_arr = call_get_accel(X_arr, M_arr_np);
        if (!A_arr) {
            Py_DECREF(X_arr);
            Py_DECREF(M_arr_np);
            Py_DECREF(results);
            return nullptr;
        }
        A = (double*)PyArray_DATA(A_arr);
        
        // V(t+dt) = V(t+dt/2) + A(t+dt) * dt/2
        for (int i = 0; i < N * 3; i++) {
            V[i] += A[i] * dt / 2.0;
        }
        
        // Compute energies
        std::vector<double> KE(N);
        std::vector<double> PE(N, 0.0);
        
        for (int i = 0; i < N; i++) {
            double v2 = V[i*3+0]*V[i*3+0] + V[i*3+1]*V[i*3+1] + V[i*3+2]*V[i*3+2];
            KE[i] = 0.5 * M[i] * v2;
        }
        
        for (int i = 0; i < N; i++) {
            for (int j = i+1; j < N; j++) {
                double dx = X[i*3+0] - X[j*3+0];
                double dy = X[i*3+1] - X[j*3+1];
                double dz = X[i*3+2] - X[j*3+2];
                double r = std::sqrt(dx*dx + dy*dy + dz*dz);
                double pe_term = -G * M[i] * M[j] / r;
                PE[i] += pe_term;
                PE[j] += pe_term;
            }
        }
        
        // Store results
        for (int i = 0; i < N; i++) {
            int row = step * N + i;
            results_data[row*11 + 0] = sim_id;
            results_data[row*11 + 1] = t / yr;
            results_data[row*11 + 2] = i;
            results_data[row*11 + 3] = X[i*3 + 0];
            results_data[row*11 + 4] = X[i*3 + 1];
            results_data[row*11 + 5] = X[i*3 + 2];
            results_data[row*11 + 6] = V[i*3 + 0];
            results_data[row*11 + 7] = V[i*3 + 1];
            results_data[row*11 + 8] = V[i*3 + 2];
            results_data[row*11 + 9] = KE[i];
            results_data[row*11 + 10] = PE[i];
        }
    }
    
    // Clean up
    Py_DECREF(X_arr);
    Py_DECREF(M_arr_np);
    Py_DECREF(A_arr);
    
    return (PyObject*)results;
}

static PyMethodDef simulator_methods[] = {
    {"run_simulation", run_simulation, METH_VARARGS, 
     "Run a single N-body simulation with perturbations"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef simulator_module = {
    PyModuleDef_HEAD_INIT,
    "simulator",
    "N-body simulation integration module",
    -1,
    simulator_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyMODINIT_FUNC PyInit_simulator(void) {
    import_array();
    return PyModule_Create(&simulator_module);
}