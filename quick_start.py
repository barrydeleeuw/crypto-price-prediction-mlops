#!/usr/bin/env python3
"""
Quick start script for the Crypto Price Prediction MLOps Pipeline.
This script demonstrates the complete pipeline from data loading to API deployment.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Failed!")
        print(f"Error: {e}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False

def main():
    """Main quick start function."""
    print("🚀 Crypto Price Prediction MLOps Pipeline - Quick Start")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Install dependencies
    print("\n📦 Step 1: Installing dependencies")
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Step 2: Run ETL pipeline
    print("\n🔄 Step 2: Running ETL pipeline")
    if not run_command("python src/run_etl.py", "Running data extraction and preprocessing"):
        print("❌ Failed to run ETL pipeline")
        sys.exit(1)
    
    # Step 3: Train model
    print("\n🤖 Step 3: Training model")
    if not run_command("python src/run_training.py", "Training the machine learning model"):
        print("❌ Failed to train model")
        sys.exit(1)
    
    # Step 4: Start API server
    print("\n🌐 Step 4: Starting API server")
    print("Starting FastAPI server in the background...")
    
    # Start the server in the background
    try:
        server_process = subprocess.Popen(
            ["python", "src/app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for the server to start
        time.sleep(5)
        
        # Check if server is running
        if server_process.poll() is None:
            print("✅ API server started successfully!")
            print("🌐 API Documentation: http://127.0.0.1:8000/docs")
            print("🏥 Health Check: http://127.0.0.1:8000/health")
            print("\n📝 Example API usage:")
            print("""
curl -X POST "http://127.0.0.1:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "SNo": 1,
       "Name": "Bitcoin",
       "Symbol": "BTC",
       "Date": "2023-01-01",
       "High": 45000.0,
       "Low": 44000.0,
       "Open": 44500.0,
       "Volume": 1000000.0,
       "Marketcap": 850000000000.0
     }'
            """)
            
            print("\n⏹️  To stop the server, press Ctrl+C")
            
            try:
                # Keep the server running
                server_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                server_process.terminate()
                server_process.wait()
                print("✅ Server stopped")
                
        else:
            print("❌ Failed to start API server")
            stdout, stderr = server_process.communicate()
            if stdout:
                print("Stdout:", stdout)
            if stderr:
                print("Stderr:", stderr)
                
    except Exception as e:
        print(f"❌ Error starting server: {e}")
    
    print("\n🎉 Quick start completed!")
    print("\n📚 Next steps:")
    print("1. Explore the API documentation at http://127.0.0.1:8000/docs")
    print("2. Try making predictions using the API")
    print("3. Check the MLflow UI for experiment tracking")
    print("4. Review the generated model artifacts in the models/ directory")

if __name__ == "__main__":
    main() 