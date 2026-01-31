#!/bin/bash

# SocialSenseAR AWS Deployment Script
# This script automates the deployment of your application to AWS EC2

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="socialsensear"
AWS_REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
AMI_ID=""  # Will be set based on region
ECR_REPO_NAME="${APP_NAME}"
KEY_NAME="${APP_NAME}-key"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SocialSenseAR AWS Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not found. Please install it first.${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not found. Please install it first.${NC}"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: AWS credentials not configured. Run 'aws configure' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites check passed${NC}"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  AWS Account: ${AWS_ACCOUNT_ID}"
echo "  Region: ${AWS_REGION}"
echo "  Instance Type: ${INSTANCE_TYPE}"
echo "  ECR Repository: ${ECR_URI}"

# Create ECR repository if it doesn't exist
echo -e "\n${YELLOW}Setting up ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${AWS_REGION} &> /dev/null; then
    aws ecr create-repository \
        --repository-name ${ECR_REPO_NAME} \
        --region ${AWS_REGION} \
        --image-scanning-configuration scanOnPush=true
    echo -e "${GREEN}✓ ECR repository created${NC}"
else
    echo -e "${GREEN}✓ ECR repository already exists${NC}"
fi

# Login to ECR
echo -e "\n${YELLOW}Logging into ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
echo -e "${GREEN}✓ Logged into ECR${NC}"

# Build Docker image
echo -e "\n${YELLOW}Building Docker image...${NC}"
# IMPORTANT: Most Macs are arm64; EC2 g4dn is amd64.
# Force an amd64 image so it can run on the EC2 instance.
if docker buildx version >/dev/null 2>&1; then
    docker buildx build --platform linux/amd64 -t ${APP_NAME}:latest --load .
else
    docker build --platform linux/amd64 -t ${APP_NAME}:latest .
fi
echo -e "${GREEN}✓ Docker image built${NC}"

# Tag and push to ECR
echo -e "\n${YELLOW}Pushing image to ECR...${NC}"
docker tag ${APP_NAME}:latest ${ECR_URI}:latest
TIMESTAMP_TAG="$(date +%Y%m%d-%H%M%S)"
docker tag ${APP_NAME}:latest ${ECR_URI}:${TIMESTAMP_TAG}
docker push ${ECR_URI}:latest
docker push ${ECR_URI}:${TIMESTAMP_TAG}
echo -e "${GREEN}✓ Image pushed to ECR${NC}"

# Store secrets in AWS Secrets Manager
echo -e "\n${YELLOW}Setting up secrets in AWS Secrets Manager...${NC}"
SECRET_NAME="${APP_NAME}-secrets"

# Check if .env file exists
if [ -f .env ]; then
    echo "Found .env file. Creating/updating secret..."

    # Read .env and convert to JSON
    SECRET_STRING=$(cat .env | grep -v '^#' | grep -v '^$' | awk -F= '{print "\""$1"\":\""$2"\""}' | paste -sd, - | sed 's/^/{/' | sed 's/$/}/')

    # Create or update secret
    if aws secretsmanager describe-secret --secret-id ${SECRET_NAME} --region ${AWS_REGION} &> /dev/null; then
        aws secretsmanager update-secret \
            --secret-id ${SECRET_NAME} \
            --secret-string "${SECRET_STRING}" \
            --region ${AWS_REGION}
        echo -e "${GREEN}✓ Secret updated${NC}"
    else
        aws secretsmanager create-secret \
            --name ${SECRET_NAME} \
            --secret-string "${SECRET_STRING}" \
            --region ${AWS_REGION}
        echo -e "${GREEN}✓ Secret created${NC}"
    fi
else
    echo -e "${YELLOW}Warning: .env file not found. You'll need to set up secrets manually.${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment preparation complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Run the EC2 setup script to create your instance:"
echo "   ./aws_ec2_setup.sh"
echo ""
echo "2. Or manually create an EC2 instance with:"
echo "   - AMI: Deep Learning AMI GPU PyTorch (Ubuntu 22.04)"
echo "   - Instance Type: ${INSTANCE_TYPE}"
echo "   - Security Group: Allow SSH (22) and your application ports"
echo "   - IAM Role: With ECR pull, Secrets Manager read permissions"
echo ""
echo "3. SSH into your instance and run:"
echo "   ./aws_run_container.sh"
