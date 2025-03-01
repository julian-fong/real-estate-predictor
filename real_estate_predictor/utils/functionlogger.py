import pickle
import functools
import pandas as pd

class FunctionLogger:
    def __init__(self):
        """Initialize a logger instance with an empty log list and pending call list."""
        self.function_logs = []
        self.pending_calls = []  # Store calls with defer=True

    @staticmethod
    def log_function_call(func):
        """Decorator to log function name, parameters, function reference, and defer."""
        @functools.wraps(func)
        def wrapper(instance, *args, **kwargs):
            defer = kwargs.pop('defer', False)  # Default to False if defer is not provided
            log_entry = {
                "function": func.__name__,
                "args": args,
                "kwargs": kwargs,
                "defer": defer
            }
            
            # If defer is False, execute the function
            if not defer:
                result = func(instance, *args, **kwargs)
                instance.function_logs.append(log_entry)
                return result
            else:
                # If defer is True, store the call in pending_calls
                instance.pending_calls.append(log_entry)
                return None  # No execution, just logging
            
        return wrapper

    def execute_pending_calls(self, clear_pending_calls = False):
        """Execute all pending function calls in sequence."""
        df = getattr(self, 'df', None)
        for log_entry in self.pending_calls:
            func_name = log_entry["function"]
            args = log_entry["args"]
            kwargs = log_entry["kwargs"]
            print(f"Executing pending function: {func_name} with args={args} kwargs={kwargs}")
            
            # Call the function dynamically within the class
            func = getattr(self, func_name, None)
            if func and isinstance(df, pd.DataFrame):
                df = func(*args, **kwargs)  # Execute the function with stored arguments
                #func(*args, **kwargs)  # Execute the function with stored arguments
            else:
                print(f"Function {func_name} or df not found!")

        # Clear pending calls after execution
        if clear_pending_calls:
            self.pending_calls = []
            
        return df
    
    def execute_logged_calls(self, clear_logged_calls = False):
        """Execute all logged function calls in sequence."""
        df = getattr(self, 'df', None)
        if not isinstance(df, pd.DataFrame):
            print("No DataFrame found in class instance")
        for log_entry in self.function_logs:
            func_name = log_entry["function"]
            args = log_entry["args"]
            kwargs = log_entry["kwargs"]
            print(f"Executing logged function: {func_name} with args={args} kwargs={kwargs}")
            
            # Call the function dynamically within the class
            func = getattr(self, func_name, None)
            if func:
                df = func(*args, **kwargs)  # Execute the function with stored arguments
                print(df.shape)
            else:
                print(f"Function {func_name} not found!")

            self.function_logs.pop(-1)
            
        return df
     
     