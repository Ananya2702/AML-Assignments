import os
import time
import requests
import subprocess

def test_docker():
    # Build the Docker image
    subprocess.run(["docker", "build", "-t", "flask-app", "."], check=True)
    
    # Run the Docker container
    container = subprocess.Popen(["docker", "run", "-p", "5000:5000", "flask-app"])
    
    # Give the container some time to start
    time.sleep(5)
    
    try:
        # Send a request to the /score endpoint
        response = requests.post("http://localhost:5000/score", json={"text": "sample text"})
        
        # Check if the response is as expected
        assert response.status_code == 200
        assert response.json() == {"score": "expected result"}  # Replace with the actual expected result
        
    finally:
        # Stop the Docker container
        container.terminate()
        container.wait()

if __name__ == "__main__":
    test_docker()
