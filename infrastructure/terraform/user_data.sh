#!/bin/bash
# User data script for EC2 instances
# This script runs on instance launch to set up the application

set -e

# Log output
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting SocialSenseAR instance setup..."

# Update system
apt-get update
apt-get upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker ubuntu
fi

# Install nvidia-container-toolkit
echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Test GPU
nvidia-smi

# Install AWS CLI if not present
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    ./aws/install
    rm -rf aws awscliv2.zip
fi

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region ${aws_region} | \
    docker login --username AWS --password-stdin ${ecr_repository%/*}

# Pull application image
echo "Pulling application image..."
docker pull ${ecr_repository}:latest

# Get secrets from Secrets Manager
echo "Retrieving secrets..."
SECRET_JSON=$(aws secretsmanager get-secret-value \
    --secret-id socialsensear-secrets \
    --region ${aws_region} \
    --query SecretString \
    --output text)

# Create .env file
echo "$SECRET_JSON" | jq -r 'to_entries|map("\(.key)=\(.value)")|.[]' > /tmp/.env

# Stop existing container if running
docker stop socialsensear 2>/dev/null || true
docker rm socialsensear 2>/dev/null || true

# Run container
echo "Starting application container..."
docker run -d \
    --name socialsensear \
    --gpus all \
    --env-file /tmp/.env \
    --restart unless-stopped \
    -p 8000:8000 \
    -p 3478:3478 \
    -p 49152-49200:49152-49200/udp \
    -v /home/ubuntu/socialsensear-data:/app/output \
    -v /home/ubuntu/socialsensear-logs:/app/logs \
    ${ecr_repository}:latest

# Clean up secrets file
rm /tmp/.env

# Wait for container to start
sleep 10

# Check container status
docker ps --filter name=socialsensear

echo "Setup complete! Container logs:"
docker logs --tail 50 socialsensear

echo "Instance setup completed successfully!"
