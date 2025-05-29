#!/bin/bash

# Exit on error
set -e

# Variables
REPO_URL="https://github.com/KrajncAndraz/strinder-orv"  # <-- CHANGE THIS TO YOUR REPO URL
REPO_DIR="strinder-orv"                                       # <-- CHANGE THIS TO YOUR REPO NAME

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "[INFO] Docker not found. Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo usermod -aG docker $USER
    echo "[INFO] Docker installed. You may need to log out and log back in for group changes to take effect."
fi

# Clone the repository
if [ ! -d "$REPO_DIR" ]; then
    echo "[INFO] Cloning repository..."
    git clone "$REPO_URL"
fi

cd "$REPO_DIR"
sudo ufw allow 5000/tcp
# Build Docker image
echo "[INFO] Building Docker image..."
sudo docker rm -f strinder-orv 2>/dev/null || echo "Container not found or already removed."
sudo docker build -f docker/Dockerfile -t strinder-orv .

# Run Docker container
echo "[INFO] Running Docker container on port 5000..."
sudo docker run -p 5000:5000 strinder-orv
echo "[INFO] Setup complete. Flask app is running on http://localhost:5000"