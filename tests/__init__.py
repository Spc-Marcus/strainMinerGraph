import os
import importlib

# Import all test files in the directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.startswith('test_') and file.endswith('.py'):
        # Convert filename to module name (remove .py extension)
        module_name = file[:-3]
        # Import the module
        importlib.import_module(f'tests.{module_name}')
if __name__ == "__main__":
    import pytest
    pytest.main()
