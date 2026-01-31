#!/bin/bash

# Container Run Script for EC2 Instance
# This script should be run ON the EC2 instance to start the application

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
APP_NAME="socialsensear"
AWS_REGION="${AWS_REGION:-us-east-1}"
SECRET_NAME="${APP_NAME}-secrets"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting SocialSenseAR Container${NC}"
echo -e "${GREEN}========================================${NC}"

# Ensure jq exists (needed to turn SecretsManager JSON into env-file)
if ! command -v jq &> /dev/null; then
    echo -e "\n${YELLOW}Installing jq...${NC}"
    sudo apt-get update -y
    sudo apt-get install -y jq
fi

# Check GPU
echo -e "\n${YELLOW}Checking GPU availability...${NC}"
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: GPU not available or nvidia drivers not installed${NC}"
    exit 1
fi
nvidia-smi
echo -e "${GREEN}✓ GPU detected${NC}"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${APP_NAME}"

# Login to ECR
echo -e "\n${YELLOW}Logging into ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
echo -e "${GREEN}✓ Logged into ECR${NC}"

# Pull latest image
echo -e "\n${YELLOW}Pulling latest Docker image...${NC}"
docker pull ${ECR_URI}:latest
echo -e "${GREEN}✓ Image pulled${NC}"

# Retrieve secrets from AWS Secrets Manager
echo -e "\n${YELLOW}Retrieving secrets from AWS Secrets Manager...${NC}"
SECRET_JSON=$(aws secretsmanager get-secret-value \
    --secret-id ${SECRET_NAME} \
    --region ${AWS_REGION} \
    --query SecretString \
    --output text)

# Create .env file from secrets
echo "${SECRET_JSON}" | jq -r 'to_entries|map("\(.key)=\(.value)")|.[]' > /tmp/.env
echo -e "${GREEN}✓ Secrets retrieved${NC}"

# Stop existing container if running
echo -e "\n${YELLOW}Stopping existing container if running...${NC}"
docker stop ${APP_NAME} 2>/dev/null || true
docker rm ${APP_NAME} 2>/dev/null || true

# Run container
echo -e "\n${YELLOW}Starting container...${NC}"
docker run -d \
    --name ${APP_NAME} \
    --gpus all \
    --env-file /tmp/.env \
    --restart unless-stopped \
    -v /home/ubuntu/socialsensear-data:/app/output \
    -v /home/ubuntu/socialsensear-logs:/app/logs \
    ${ECR_URI}:latest

# Clean up secrets file
rm /tmp/.env

echo -e "${GREEN}✓ Container started${NC}"

# Wait a moment for container to start
sleep 5

# Check container status
echo -e "\n${YELLOW}Container status:${NC}"
docker ps --filter name=${APP_NAME}

# Show logs
echo -e "\n${YELLOW}Recent logs:${NC}"
docker logs --tail 50 ${APP_NAME}

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Container is running!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Useful commands:${NC}"
echo "  View logs:        docker logs -f ${APP_NAME}"
echo "  Stop container:   docker stop ${APP_NAME}"
echo "  Start container:  docker start ${APP_NAME}"
echo "  Restart:          docker restart ${APP_NAME}"
echo "  Shell access:     docker exec -it ${APP_NAME} bash"
echo "  GPU monitoring:   watch -n 1 nvidia-smi"
echo ""
echo "  Output files:     ~/socialsensear-data/"
echo "  Log files:        ~/socialsensear-logs/"
