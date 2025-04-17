import subprocess
import time
import sys
import os

def run_backend():
    """Run the backend server"""
    print("Starting backend server...")
    backend_process = subprocess.Popen(
        [sys.executable, "backend/server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # Give the server a moment to start
    time.sleep(2)
    return backend_process

def run_frontend():
    """Run the frontend application"""
    print("Starting frontend application...")
    frontend_process = subprocess.Popen(
        [sys.executable, "frontend/road_network_simulator.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return frontend_process

def main():
    """Main function to run the application"""
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Start the backend
    backend_process = run_backend()
    
    # Start the frontend
    frontend_process = run_frontend()
    
    try:
        # Wait for the frontend to finish
        frontend_process.wait()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Clean up
        if backend_process.poll() is None:
            backend_process.terminate()
        if frontend_process.poll() is None:
            frontend_process.terminate()
    
    print("Application closed.")

if __name__ == "__main__":
    main()
