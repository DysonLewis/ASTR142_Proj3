/*
N-body simulation integration module in C++
Handles the main simulation loop with leapfrog integration
Calls Python accel module for gravitational acceleration calculations
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <vector>
#include <omp.h>
#include <deque>

#define G 6.6743e-8L  // gravitational constant in cm^3 g^-1 s^-2

struct SimulationRecord {
    int simulation;
    double time_yr;
    int body_idx;
    double x, y, z;
    double vx, vy, vz;
    double KE, PE, E_tot;
};

/*
Call the Python accel module's get_accel function to compute accelerations.

Args:
    X_arr: N x 3 numpy array of positions in cm
    M_arr: N length numpy array of masses in grams
    
Returns:
    N x 3 numpy array of accelerations in cm/s^2
*/
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

/*
Handle elastic collisions between particles.
Detects when particles are closer than collision_radius and applies elastic collision physics.

Args:
    X: positions vector (flattened)
    V: velocities vector (flattened)
    M: masses vector
    N: number of particles
    collision_radius: minimum distance before collision handling
*/
static void handle_collisions(std::vector<double>& X, std::vector<double>& V, 
                             const std::vector<double>& M, int N, double collision_radius) {
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            double dx = X[i*3+0] - X[j*3+0];
            double dy = X[i*3+1] - X[j*3+1];
            double dz = X[i*3+2] - X[j*3+2];
            double r = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            if (r < collision_radius) {
                // Collision normal vector
                double nx = dx / r;
                double ny = dy / r;
                double nz = dz / r;
                
                // Relative velocity
                double v_rel_x = V[i*3+0] - V[j*3+0];
                double v_rel_y = V[i*3+1] - V[j*3+1];
                double v_rel_z = V[i*3+2] - V[j*3+2];
                
                // Relative velocity along normal
                double v_rel_n = v_rel_x*nx + v_rel_y*ny + v_rel_z*nz;
                
                // Only apply collision if particles are approaching
                if (v_rel_n < 0) {
                    double m_i = M[i];
                    double m_j = M[j];
                    double m_total = m_i + m_j;
                    
                    // Elastic collision impulse
                    double impulse = 2.0 * m_i * m_j * v_rel_n / m_total;
                    
                    // Apply impulse to velocities
                    V[i*3+0] -= (impulse / m_i) * nx;
                    V[i*3+1] -= (impulse / m_i) * ny;
                    V[i*3+2] -= (impulse / m_i) * nz;
                    
                    V[j*3+0] += (impulse / m_j) * nx;
                    V[j*3+1] += (impulse / m_j) * ny;
                    V[j*3+2] += (impulse / m_j) * nz;
                    
                    // Separate overlapping particles
                    double overlap = collision_radius - r;
                    double separation = overlap / 2.0 + 1e-10;
                    
                    X[i*3+0] += separation * nx;
                    X[i*3+1] += separation * ny;
                    X[i*3+2] += separation * nz;
                    
                    X[j*3+0] -= separation * nx;
                    X[j*3+1] -= separation * ny;
                    X[j*3+2] -= separation * nz;
                }
            }
        }
    }
}

/*
Check if system has reached virial equilibrium using moving average.
Virial theorem: 2*KE + PE ≈ 0 at equilibrium
Returns true if virial ratio is stable near 1.0

Args:
    virial_ratios: deque of recent virial ratio measurements
    window_size: number of measurements to average
    tolerance: acceptable deviation from ideal ratio
*/
static bool check_virial_equilibrium(const std::deque<double>& virial_ratios, 
                                    int window_size, double tolerance) {
    if ((int)virial_ratios.size() < window_size) {
        return false;
    }
    
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = (int)virial_ratios.size() - window_size; i < (int)virial_ratios.size(); i++) {
        sum += virial_ratios[i];
        sum_sq += virial_ratios[i] * virial_ratios[i];
    }
    
    double mean = sum / window_size;
    double variance = (sum_sq / window_size) - (mean * mean);
    double std_dev = std::sqrt(variance);
    
    return (std::abs(mean - 1.0) < tolerance) && (std_dev < tolerance);
}

/*
Run a single N-body simulation using leapfrog integration.

Args:
    X0_arr: N x 3 initial positions in cm
    V0_arr: N x 3 initial velocities in cm/s
    M_arr: N masses in grams
    perturb_idx_arr: unused (kept for compatibility)
    perturb_pos_arr: unused (kept for compatibility)
    perturb_vel_arr: unused (kept for compatibility)
    sim_id: simulation identifier
    max_step: maximum number of timesteps
    dt: timestep in seconds
    yr: conversion factor from seconds to years
    collision_radius: minimum distance for collision detection
    
Returns:
    (actual_steps*N) x 11 numpy array containing:
    [sim_id, time_yr, body_idx, x, y, z, vx, vy, vz, KE, PE]
*/
static PyObject* run_simulation([[maybe_unused]] PyObject* self, PyObject* args) {
    PyArrayObject* X0_arr = nullptr;
    PyArrayObject* V0_arr = nullptr;
    PyArrayObject* M_arr = nullptr;
    PyArrayObject* perturb_idx_arr = nullptr;
    PyArrayObject* perturb_pos_arr = nullptr;
    PyArrayObject* perturb_vel_arr = nullptr;
    
    int sim_id, max_step;
    double dt, yr, collision_radius;
    
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!iiddd",
                          &PyArray_Type, &X0_arr,
                          &PyArray_Type, &V0_arr,
                          &PyArray_Type, &M_arr,
                          &PyArray_Type, &perturb_idx_arr,
                          &PyArray_Type, &perturb_pos_arr,
                          &PyArray_Type, &perturb_vel_arr,
                          &sim_id,
                          &max_step,
                          &dt,
                          &yr,
                          &collision_radius)) {
        return nullptr;
    }
    
    int N = PyArray_DIM(X0_arr, 0);
    
    // Copy initial conditions to C++ vectors for fast access
    std::vector<double> X(N * 3);  // positions flattened [x0,y0,z0,x1,y1,z1,...]
    std::vector<double> V(N * 3);  // velocities flattened
    std::vector<double> M(N);      // masses
    
    double* X0_data = (double*)PyArray_DATA(X0_arr);
    double* V0_data = (double*)PyArray_DATA(V0_arr);
    double* M_data = (double*)PyArray_DATA(M_arr);
    
    // Parallelize initial data copying
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N * 3; i++) {
        X[i] = X0_data[i];
        V[i] = V0_data[i];
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        M[i] = M_data[i];
    }
    
    // Allocate temporary storage for results (will be trimmed later)
    std::vector<std::vector<double>> results_buffer;
    results_buffer.reserve(max_step * N);
    
    // Create numpy arrays for passing to accel module
    npy_intp pos_dims[2] = {N, 3};
    npy_intp mass_dims[1] = {N};
    
    PyArrayObject* X_arr = (PyArrayObject*)PyArray_SimpleNew(2, pos_dims, NPY_DOUBLE);
    PyArrayObject* M_arr_np = (PyArrayObject*)PyArray_SimpleNew(1, mass_dims, NPY_DOUBLE);
    
    if (!X_arr || !M_arr_np) {
        Py_XDECREF(X_arr);
        Py_XDECREF(M_arr_np);
        return nullptr;
    }
    
    // Copy masses to numpy array (only need to do once since masses don't change)
    double* M_arr_data = (double*)PyArray_DATA(M_arr_np);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        M_arr_data[i] = M[i];
    }
    
    // Copy initial positions and compute initial acceleration
    double* X_arr_data = (double*)PyArray_DATA(X_arr);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N * 3; i++) {
        X_arr_data[i] = X[i];
    }
    
    PyArrayObject* A_arr = call_get_accel(X_arr, M_arr_np);
    if (!A_arr) {
        Py_DECREF(X_arr);
        Py_DECREF(M_arr_np);
        return nullptr;
    }
    
    double* A = (double*)PyArray_DATA(A_arr);
    
    // Virial equilibrium tracking
    std::deque<double> virial_ratios;
    const int check_interval = 100;     // how often it checks for stability
    const int window_size = 200;        // how many step it averages over for stability to be true
    const double tolerance = 0.05;      // precent deviation
    const int min_steps = 1000;         // minimum runtime before checking
    bool equilibrium_reached = false;
    int actual_steps = 0;
    
    // Main leapfrog integration loop
    for (int step = 0; step < max_step; step++) {
        double t = step * dt;
        
        // Leapfrog step 1: V(t+dt/2) = V(t) + A(t)*dt/2
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N * 3; i++) {
            V[i] += A[i] * dt / 2.0;
        }
        
        // Leapfrog step 2: X(t+dt) = X(t) + V(t+dt/2)*dt
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N * 3; i++) {
            X[i] += V[i] * dt;
        }
        
        // Handle particle collisions
        handle_collisions(X, V, M, N, collision_radius);
        
        // Update positions in numpy array for accel calculation
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N * 3; i++) {
            X_arr_data[i] = X[i];
        }
        
        // Compute A(t+dt) at new positions
        Py_DECREF(A_arr);
        A_arr = call_get_accel(X_arr, M_arr_np);
        if (!A_arr) {
            Py_DECREF(X_arr);
            Py_DECREF(M_arr_np);
            return nullptr;
        }
        A = (double*)PyArray_DATA(A_arr);
        
        // Leapfrog step 3: V(t+dt) = V(t+dt/2) + A(t+dt)*dt/2
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N * 3; i++) {
            V[i] += A[i] * dt / 2.0;
        }
        
        // Compute kinetic energy for each particle: KE = 0.5*m*v^2
        std::vector<double> KE(N);
        std::vector<double> PE(N, 0.0);
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            double v2 = V[i*3+0]*V[i*3+0] + V[i*3+1]*V[i*3+1] + V[i*3+2]*V[i*3+2];
            KE[i] = 0.5 * M[i] * v2;
        }
        
        // Compute potential energy for each particle
        // PE_ij = -G*m_i*m_j / r_ij
        // Each particle gets half of each pair interaction to avoid double counting when summing
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N; i++) {
            for (int j = i+1; j < N; j++) {
                double dx = X[i*3+0] - X[j*3+0];
                double dy = X[i*3+1] - X[j*3+1];
                double dz = X[i*3+2] - X[j*3+2];
                double r = std::sqrt(dx*dx + dy*dy + dz*dz);
                double pe_term = -G * M[i] * M[j] / r;
                #pragma omp atomic
                PE[i] += pe_term * 0.5;
                #pragma omp atomic
                PE[j] += pe_term * 0.5;
            }
        }
        
        // Store results for this timestep
        for (int i = 0; i < N; i++) {
            std::vector<double> row(11);
            row[0] = sim_id;
            row[1] = t / yr;
            row[2] = i;
            row[3] = X[i*3 + 0];
            row[4] = X[i*3 + 1];
            row[5] = X[i*3 + 2];
            row[6] = V[i*3 + 0];
            row[7] = V[i*3 + 1];
            row[8] = V[i*3 + 2];
            row[9] = KE[i];
            row[10] = PE[i];
            results_buffer.push_back(row);
        }
        
        actual_steps = step + 1;
        
        // Check virial equilibrium every check_interval steps
        if (step % check_interval == 0 && step >= min_steps) {
            double total_KE = 0.0;
            double total_PE = 0.0;
            
            for (int i = 0; i < N; i++) {
                total_KE += KE[i];
                total_PE += PE[i];
            }
            
            if (std::abs(total_PE) > 1e-30) {
                double virial_ratio = std::abs(2.0 * total_KE / total_PE);
                virial_ratios.push_back(virial_ratio);
                
                if (virial_ratios.size() > (size_t)(window_size * 2)) {
                    virial_ratios.pop_front();
                }
                
                if (check_virial_equilibrium(virial_ratios, window_size, tolerance)) {
                    equilibrium_reached = true;
                    break;
                }
            }
        }
    }
    
    // Clean up numpy arrays
    Py_DECREF(X_arr);
    Py_DECREF(M_arr_np);
    Py_DECREF(A_arr);
    
    // Create output array with actual size
    npy_intp dims[2] = {(npy_intp)(actual_steps * N), 11};
    PyArrayObject* results = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (!results) {
        return nullptr;
    }
    double* results_data = (double*)PyArray_DATA(results);
    
    // Copy buffered results to output array
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < actual_steps * N; i++) {
        for (int j = 0; j < 11; j++) {
            results_data[i*11 + j] = results_buffer[i][j];
        }
    }
    
    if (equilibrium_reached) {
        printf("Virial equilibrium reached at step %d (%.2f years)\n", 
               actual_steps, actual_steps * dt / yr);
    } else {
        printf("Maximum steps reached without achieving virial equilibrium\n");
    }
    
    return (PyObject*)results;
}

// Python module method definitions
static PyMethodDef simulator_methods[] = {
    {"run_simulation", run_simulation, METH_VARARGS, 
     "Run a single N-body simulation"},
    {nullptr, nullptr, 0, nullptr}
};

// Python module definition
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

// Module initialization function called by Python
PyMODINIT_FUNC PyInit_simulator(void) {
    import_array();  // required for numpy C API
    return PyModule_Create(&simulator_module);
}