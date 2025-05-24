# run.py

import os
from app import app # Assuming your Flask app instance is named 'app' in app/__init__.py or app/app.py
# We will define the actual Flask app instance in app/__init__.py or app/app.py later.

# Load configuration from config.py
# The from_object method looks for configuration variables (uppercase)
# within the specified module.
try:
    app.config.from_object('config')
    # You could also use from_pyfile if config.py was in the same directory as run.py
    # app.config.from_pyfile('config.py')
except ImportError:
    print("Error: config.py not found. Please create a config.py file.")
    # You might want to exit here or raise an exception in a real application
    pass # For now, just print an error and continue (might fail later)

# You might also want to load environment variables here
# For example, to get secret keys from Azure App Service or environment
# app.config.from_envvar('YOUR_APP_CONFIG_ENV_VAR', silent=True)

# The standard way to run a Python script as an executable
if __name__ == '__main__':
    # Get host and port from config or environment variables, default otherwise
    host = app.config.get('HOST', '127.0.0.1') # Default to localhost
    port = int(os.environ.get('PORT', app.config.get('PORT', 5000))) # Default to 5000

    # Run the Flask development server
    # debug=True provides a debugger and automatic reloading on code changes
    # Set debug=False for production environments
    debug_mode = app.config.get('DEBUG', True) # Default to True for dev

    print(f"Starting Flask server on http://{host}:{port}/")
    print(f"Debug mode: {debug_mode}")

    app.run(host=host, port=port, debug=debug_mode)