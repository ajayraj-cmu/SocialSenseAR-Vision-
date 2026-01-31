#!/bin/bash

# EC2 Instance Setup Script for SocialSenseAR
# This script creates and configures an EC2 GPU instance

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
APP_NAME="socialsensear"
AWS_REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
KEY_NAME="${APP_NAME}-key"
SECURITY_GROUP_NAME="${APP_NAME}-sg"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}EC2 Instance Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Get the latest Deep Learning GPU AMI (PyTorch + NVIDIA drivers)
echo -e "\n${YELLOW}Finding latest Deep Learning GPU AMI...${NC}"
#
# Notes:
# - Deep Learning AMIs are typically published under AWS account 898082745236 (Deep Learning AMI team),
#   not the literal "amazon" owner alias.
# - Names have changed over time; use a broad wildcard search and then pick the newest.
AMI_ID=$(aws ec2 describe-images \
    --owners 898082745236 \
    --filters "Name=name,Values=*Deep Learning*GPU*PyTorch*(Ubuntu 22.04)*" \
              "Name=state,Values=available" \
    --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' \
    --output text \
    --region ${AWS_REGION})

echo "Using AMI: ${AMI_ID}"

if [ -z "${AMI_ID}" ] || [ "${AMI_ID}" = "None" ]; then
    echo -e "${RED}Error: Could not find a Deep Learning GPU AMI in region ${AWS_REGION}.${NC}"
    echo -e "${YELLOW}Fix: Open AWS Console → EC2 → AMIs and search for 'Deep Learning AMI GPU PyTorch' in ${AWS_REGION}, or try a different region.${NC}"
    exit 1
fi

# Create key pair if it doesn't exist
echo -e "\n${YELLOW}Setting up SSH key pair...${NC}"
if ! aws ec2 describe-key-pairs --key-names ${KEY_NAME} --region ${AWS_REGION} &> /dev/null; then
    aws ec2 create-key-pair \
        --key-name ${KEY_NAME} \
        --query 'KeyMaterial' \
        --output text \
        --region ${AWS_REGION} > ${KEY_NAME}.pem
    chmod 400 ${KEY_NAME}.pem
    echo -e "${GREEN}✓ Key pair created: ${KEY_NAME}.pem${NC}"
    echo -e "${YELLOW}IMPORTANT: Save this file! It cannot be recovered.${NC}"
else
    echo -e "${GREEN}✓ Key pair already exists${NC}"
fi

# Create security group
echo -e "\n${YELLOW}Setting up security group...${NC}"
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text --region ${AWS_REGION})

if ! aws ec2 describe-security-groups --group-names ${SECURITY_GROUP_NAME} --region ${AWS_REGION} &> /dev/null 2>&1; then
    SECURITY_GROUP_ID=$(aws ec2 create-security-group \
        --group-name ${SECURITY_GROUP_NAME} \
        --description "Security group for SocialSenseAR" \
        --vpc-id ${VPC_ID} \
        --query 'GroupId' \
        --output text \
        --region ${AWS_REGION})

    # Add SSH rule
    aws ec2 authorize-security-group-ingress \
        --group-id ${SECURITY_GROUP_ID} \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region ${AWS_REGION}

    echo -e "${GREEN}✓ Security group created: ${SECURITY_GROUP_ID}${NC}"
else
    SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
        --group-names ${SECURITY_GROUP_NAME} \
        --query 'SecurityGroups[0].GroupId' \
        --output text \
        --region ${AWS_REGION})
    echo -e "${GREEN}✓ Security group already exists: ${SECURITY_GROUP_ID}${NC}"
fi

# Create IAM role for EC2
echo -e "\n${YELLOW}Setting up IAM role...${NC}"
ROLE_NAME="${APP_NAME}-ec2-role"
INSTANCE_PROFILE_NAME="${APP_NAME}-instance-profile"

# Create trust policy
cat > /tmp/trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create IAM role if it doesn't exist
if ! aws iam get-role --role-name ${ROLE_NAME} &> /dev/null; then
    aws iam create-role \
        --role-name ${ROLE_NAME} \
        --assume-role-policy-document file:///tmp/trust-policy.json

    # Attach necessary policies
    aws iam attach-role-policy \
        --role-name ${ROLE_NAME} \
        --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

    aws iam attach-role-policy \
        --role-name ${ROLE_NAME} \
        --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite

    aws iam attach-role-policy \
        --role-name ${ROLE_NAME} \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy

    echo -e "${GREEN}✓ IAM role created${NC}"
else
    echo -e "${GREEN}✓ IAM role already exists${NC}"
fi

# Create instance profile
if ! aws iam get-instance-profile --instance-profile-name ${INSTANCE_PROFILE_NAME} &> /dev/null; then
    aws iam create-instance-profile --instance-profile-name ${INSTANCE_PROFILE_NAME}
    aws iam add-role-to-instance-profile \
        --instance-profile-name ${INSTANCE_PROFILE_NAME} \
        --role-name ${ROLE_NAME}
    echo -e "${GREEN}✓ Instance profile created${NC}"
    # Wait for instance profile to be ready
    sleep 10
else
    echo -e "${GREEN}✓ Instance profile already exists${NC}"
fi

 # Create user data script
cat > /tmp/user-data.sh << 'EOF'
#!/bin/bash
# Update system
apt-get update
apt-get upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker ubuntu
fi

# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Test GPU
nvidia-smi

echo "EC2 instance setup complete!"
EOF

# Launch instance
echo -e "\n${YELLOW}Launching EC2 instance...${NC}"
echo "Instance Type: ${INSTANCE_TYPE}"
echo "This may take a few minutes..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ${AMI_ID} \
    --instance-type ${INSTANCE_TYPE} \
    --key-name ${KEY_NAME} \
    --security-group-ids ${SECURITY_GROUP_ID} \
    --iam-instance-profile Name=${INSTANCE_PROFILE_NAME} \
    --user-data file:///tmp/user-data.sh \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${APP_NAME}},{Key=Application,Value=SocialSenseAR}]" \
    --query 'Instances[0].InstanceId' \
    --output text \
    --region ${AWS_REGION})

echo -e "${GREEN}✓ Instance launched: ${INSTANCE_ID}${NC}"

# Wait for instance to be running
echo -e "\n${YELLOW}Waiting for instance to start...${NC}"
aws ec2 wait instance-running --instance-ids ${INSTANCE_ID} --region ${AWS_REGION}

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids ${INSTANCE_ID} \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text \
    --region ${AWS_REGION})

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Instance setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Instance Details:${NC}"
echo "  Instance ID: ${INSTANCE_ID}"
echo "  Public IP: ${PUBLIC_IP}"
echo "  Instance Type: ${INSTANCE_TYPE}"
echo "  Region: ${AWS_REGION}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Wait 2-3 minutes for instance initialization to complete"
echo ""
echo "2. Connect via SSH:"
echo "   ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "3. On the instance, create a run script:"
echo "   See aws_run_container.sh for the commands to run"
echo ""
echo "4. Alternative - Copy files and run:"
echo "   scp -i ${KEY_NAME}.pem aws_run_container.sh ubuntu@${PUBLIC_IP}:~/"
echo "   ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo "   chmod +x aws_run_container.sh"
echo "   ./aws_run_container.sh"

# Save instance info
cat > instance_info.txt << EOF
Instance ID: ${INSTANCE_ID}
Public IP: ${PUBLIC_IP}
Region: ${AWS_REGION}
Instance Type: ${INSTANCE_TYPE}
Key File: ${KEY_NAME}.pem
Security Group: ${SECURITY_GROUP_ID}

SSH Command:
ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}

Stop Instance:
aws ec2 stop-instances --instance-ids ${INSTANCE_ID} --region ${AWS_REGION}

Start Instance:
aws ec2 start-instances --instance-ids ${INSTANCE_ID} --region ${AWS_REGION}

Terminate Instance:
aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region ${AWS_REGION}
EOF

echo -e "\n${GREEN}Instance details saved to: instance_info.txt${NC}"
