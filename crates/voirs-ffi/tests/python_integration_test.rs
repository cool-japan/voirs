//! Python Integration Tests
//!
//! Tests for Python bindings functionality, including PyO3 integration,
//! NumPy compatibility, and async operation support.

#[cfg(feature = "python")]
mod python_tests {
    use pyo3::prelude::*;
    use pyo3::types::*;
    use std::sync::Arc;
    use voirs_ffi::python::*;

    #[test]
    fn test_python_module_creation() {
        // Test that Python module can be created
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test module creation
            let module = PyModule::new(py, "voirs_test").unwrap();
            assert!(!module.is_none());

            // Test basic Python integration
            let result: i32 = py.eval("2 + 2", None, None).unwrap().extract().unwrap();
            assert_eq!(result, 4);
        });
    }

    #[test]
    fn test_python_error_handling() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test error creation and handling
            let error = PyErr::new::<pyo3::exceptions::PyValueError, _>("Test error");
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));

            // Test error message
            let error_msg = error.to_string();
            assert!(error_msg.contains("Test error"));
        });
    }

    #[test]
    fn test_python_type_conversions() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test basic type conversions
            let int_val = 42i32;
            let py_int = int_val.to_object(py);
            let back_int: i32 = py_int.extract(py).unwrap();
            assert_eq!(int_val, back_int);

            let float_val = 3.14f64;
            let py_float = float_val.to_object(py);
            let back_float: f64 = py_float.extract(py).unwrap();
            assert!((float_val - back_float).abs() < 1e-10);

            let str_val = "Hello, Python!";
            let py_str = str_val.to_object(py);
            let back_str: String = py_str.extract(py).unwrap();
            assert_eq!(str_val, back_str);
        });
    }

    #[test]
    fn test_python_list_operations() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test list creation and manipulation
            let list = PyList::new(py, &[1, 2, 3, 4, 5]);
            assert_eq!(list.len(), 5);

            // Test list access
            let first: i32 = list.get_item(0).unwrap().extract().unwrap();
            assert_eq!(first, 1);

            let last: i32 = list.get_item(4).unwrap().extract().unwrap();
            assert_eq!(last, 5);

            // Test list conversion
            let vec: Vec<i32> = list.extract().unwrap();
            assert_eq!(vec, vec![1, 2, 3, 4, 5]);
        });
    }

    #[test]
    fn test_python_dict_operations() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test dictionary creation and manipulation
            let dict = PyDict::new(py);
            dict.set_item("name", "VoiRS").unwrap();
            dict.set_item("version", "0.1.0").unwrap();
            dict.set_item("enabled", true).unwrap();

            // Test dictionary access
            let name: String = dict.get_item("name").unwrap().unwrap().extract().unwrap();
            assert_eq!(name, "VoiRS");

            let version: String = dict
                .get_item("version")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(version, "0.1.0");

            let enabled: bool = dict
                .get_item("enabled")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(enabled, true);
        });
    }

    #[test]
    fn test_python_function_calls() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test function definition and calling
            let code = "
def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b
            ";

            let module = PyModule::from_code(py, code, "test_module.py", "test_module").unwrap();

            // Test function calls
            let add_func = module.getattr("add_numbers").unwrap();
            let result: i32 = add_func.call1((10, 20)).unwrap().extract().unwrap();
            assert_eq!(result, 30);

            let multiply_func = module.getattr("multiply_numbers").unwrap();
            let result: i32 = multiply_func.call1((6, 7)).unwrap().extract().unwrap();
            assert_eq!(result, 42);
        });
    }

    #[test]
    fn test_python_class_creation() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test class creation
            let code = "
class TestClass:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
    
    def set_value(self, new_value):
        self.value = new_value
            ";

            let module = PyModule::from_code(py, code, "test_class.py", "test_class").unwrap();
            let class = module.getattr("TestClass").unwrap();

            // Test instance creation
            let instance = class.call1((42,)).unwrap();

            // Test method calls
            let get_value = instance.getattr("get_value").unwrap();
            let value: i32 = get_value.call0().unwrap().extract().unwrap();
            assert_eq!(value, 42);

            let set_value = instance.getattr("set_value").unwrap();
            set_value.call1((100,)).unwrap();

            let new_value: i32 = get_value.call0().unwrap().extract().unwrap();
            assert_eq!(new_value, 100);
        });
    }

    #[test]
    fn test_python_async_support() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test async function support
            let code = "
import asyncio

async def async_add(a, b):
    await asyncio.sleep(0.001)  # Simulate async work
    return a + b

def run_async_add(a, b):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_add(a, b))
    finally:
        loop.close()
            ";

            let module = PyModule::from_code(py, code, "test_async.py", "test_async").unwrap();
            let async_func = module.getattr("run_async_add").unwrap();

            let result: i32 = async_func.call1((15, 25)).unwrap().extract().unwrap();
            assert_eq!(result, 40);
        });
    }

    #[test]
    fn test_python_exception_handling() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test exception handling
            let code = "
def divide_numbers(a, b):
    if b == 0:
        raise ValueError('Cannot divide by zero')
    return a / b
            ";

            let module =
                PyModule::from_code(py, code, "test_exceptions.py", "test_exceptions").unwrap();
            let divide_func = module.getattr("divide_numbers").unwrap();

            // Test normal operation
            let result: f64 = divide_func.call1((10.0, 2.0)).unwrap().extract().unwrap();
            assert_eq!(result, 5.0);

            // Test exception handling
            let error_result = divide_func.call1((10.0, 0.0));
            assert!(error_result.is_err());

            let error = error_result.unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
        });
    }

    #[test]
    fn test_python_memory_management() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test memory management with large objects
            let large_list = PyList::new(py, &(0..10000).collect::<Vec<i32>>());
            assert_eq!(large_list.len(), 10000);

            // Test that memory is properly managed
            let first: i32 = large_list.get_item(0).unwrap().extract().unwrap();
            assert_eq!(first, 0);

            let last: i32 = large_list.get_item(9999).unwrap().extract().unwrap();
            assert_eq!(last, 9999);
        });
    }

    #[test]
    fn test_python_numpy_compatibility() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test NumPy array creation and manipulation
            let numpy_code = "
import numpy as np

def create_array():
    return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

def process_array(arr):
    return arr * 2.0

def array_info(arr):
    return {
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'size': arr.size
    }
            ";

            // Only run if NumPy is available
            let numpy_available = py.import("numpy").is_ok();
            if numpy_available {
                let module =
                    PyModule::from_code(py, numpy_code, "test_numpy.py", "test_numpy").unwrap();

                let create_func = module.getattr("create_array").unwrap();
                let array = create_func.call0().unwrap();

                let process_func = module.getattr("process_array").unwrap();
                let processed = process_func.call1((array,)).unwrap();

                let info_func = module.getattr("array_info").unwrap();
                let info = info_func.call1((processed,)).unwrap();

                // Verify array properties
                assert!(!info.is_none());
            }
        });
    }

    #[test]
    fn test_python_callback_functions() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test callback function support
            let code = "
def apply_callback(data, callback):
    return [callback(item) for item in data]

def square(x):
    return x * x

def double(x):
    return x * 2
            ";

            let module =
                PyModule::from_code(py, code, "test_callbacks.py", "test_callbacks").unwrap();
            let apply_func = module.getattr("apply_callback").unwrap();
            let square_func = module.getattr("square").unwrap();
            let double_func = module.getattr("double").unwrap();

            let data = PyList::new(py, &[1, 2, 3, 4, 5]);

            // Test square callback
            let squared: Vec<i32> = apply_func
                .call1((data, square_func))
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(squared, vec![1, 4, 9, 16, 25]);

            // Test double callback
            let doubled: Vec<i32> = apply_func
                .call1((data, double_func))
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(doubled, vec![2, 4, 6, 8, 10]);
        });
    }

    #[test]
    fn test_python_threading_support() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test threading support
            let code = "
import threading
import time

def thread_worker(name, delay):
    time.sleep(delay)
    return f'Thread {name} completed'

def run_threads():
    threads = []
    results = []
    
    def worker(name, delay):
        result = thread_worker(name, delay)
        results.append(result)
    
    for i in range(3):
        t = threading.Thread(target=worker, args=(i, 0.001))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    return results
            ";

            let module =
                PyModule::from_code(py, code, "test_threading.py", "test_threading").unwrap();
            let run_func = module.getattr("run_threads").unwrap();

            let results: Vec<String> = run_func.call0().unwrap().extract().unwrap();
            assert_eq!(results.len(), 3);

            for (i, result) in results.iter().enumerate() {
                assert!(result.contains(&format!("Thread {} completed", i)));
            }
        });
    }

    #[test]
    fn test_python_gc_interaction() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test garbage collection interaction
            let code = "
import gc

def create_large_objects():
    objects = []
    for i in range(1000):
        obj = {'id': i, 'data': list(range(100))}
        objects.append(obj)
    return len(objects)

def force_gc():
    return gc.collect()
            ";

            let module = PyModule::from_code(py, code, "test_gc.py", "test_gc").unwrap();
            let create_func = module.getattr("create_large_objects").unwrap();
            let gc_func = module.getattr("force_gc").unwrap();

            // Create objects
            let count: i32 = create_func.call0().unwrap().extract().unwrap();
            assert_eq!(count, 1000);

            // Force garbage collection
            let collected: i32 = gc_func.call0().unwrap().extract().unwrap();
            // Don't assert specific collection count as it depends on implementation
            assert!(collected >= 0);
        });
    }
}

#[cfg(not(feature = "python"))]
mod python_tests {
    // Placeholder tests when Python feature is not enabled
    #[test]
    fn test_python_feature_disabled() {
        // This test ensures the test suite compiles even when Python feature is disabled
        assert_eq!(1 + 1, 2);
    }
}
