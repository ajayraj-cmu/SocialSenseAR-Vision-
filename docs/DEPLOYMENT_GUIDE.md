# SocialSenseAR Deployment Guide

Complete guide for deploying the SocialSenseAR compute offloading infrastructure to AWS.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Production Deployment](#production-deployment)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Scaling](#scaling)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers deploying SocialSenseAR to AWS with:

- **GPU Compute**: g4dn.xlarge instances (NVIDIA T4 GPUs)
- **Auto-Scaling**: 1-10 instances based on load
- **Load Balancing**: Application Load Balancer with WebSocket support
- **Session Storage**: DynamoDB for distributed state
- **Monitoring**: CloudWatch metrics and logs
- **Zero Hardware Dependency**: All AR headset compute offloaded to cloud

**Architecture:**
```
Internet → ALB → [EC2 Instance 1, EC2 Instance 2, ...] → DynamoDB
                     ↓
                 CloudWatch
```

---

## Prerequisites

### 1. AWS Account Setup

1. **Create AWS Account**: https://aws.amazon.com/
2. **Add payment method**
3. **Request GPU instance limit increase**:
   - Go to: AWS Console → Service Quotas
   - Search for: "Running On-Demand G instances"
   - Request increase to at least 4 vCPUs (1x g4dn.xlarge)

### 2. Local Machine Setup

**Install AWS CLI:**
```bash
# macOS
brew install awscli

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Windows
# Download from: https://aws.amazon.com/cli/
```

**Configure AWS credentials:**
```bash
aws configure
```

Enter:
- AWS Access Key ID: (from IAM Console)
- AWS Secret Access Key: (from IAM Console)
- Default region: `us-east-1`
- Default output format: `json`

**Install Docker Desktop:**
- Download from: https://www.docker.com/products/docker-desktop
- Install and start Docker

**Install Terraform (for production):**
```bash
# macOS
brew install terraform

# Linux
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Verify
terraform --version
```

---

## Quick Start

### Development Deployment (Single Instance)

This is the fastest way to get started with a single EC2 instance.

**Step 1: Build and push Docker image**

```bash
cd /path/to/Meta X SocialSense
chmod +x aws_deploy.sh aws_ec2_setup.sh aws_run_container.sh

./aws_deploy.sh
```

**What this does:**
- Creates ECR repository
- Builds Docker image with GPU support
- Pushes image to AWS
- Stores API keys in Secrets Manager

**Expected time:** 10-15 minutes

**Step 2: Create GPU instance**

```bash
./aws_ec2_setup.sh
```

**What this does:**
- Creates VPC security group
- Generates SSH key pair (`socialsensear-key.pem`)
- Launches g4dn.xlarge instance
- Installs Docker + NVIDIA drivers

**Expected time:** 5-10 minutes

**Output:**
```
Instance ID: i-0123456789abcdef0
Public IP: 54.123.45.67
SSH Command: ssh -i socialsensear-key.pem ubuntu@54.123.45.67
```

**Save this information!** It's written to `instance_info.txt`.

**Step 3: Connect and start server**

```bash
# Wait 2-3 minutes for instance initialization, then:
ssh -i socialsensear-key.pem ubuntu@54.123.45.67

# On the instance, copy the run script:
# (Exit SSH first)
scp -i socialsensear-key.pem aws_run_container.sh ubuntu@54.123.45.67:~/

# Reconnect
ssh -i socialsensear-key.pem ubuntu@54.123.45.67

# Run
chmod +x aws_run_container.sh
./aws_run_container.sh
```

**Step 4: Test connection**

```bash
# From your local machine:
curl http://54.123.45.67:8000/health

# Should return:
# {"status":"healthy","gpu_available":true}
```

**Step 5: Connect client**

```bash
python client_sdk/python_example.py \
    --server-url ws://54.123.45.67:8000 \
    --api-key YOUR_API_KEY
```

---

## Production Deployment

### Using Terraform (Recommended)

Terraform provides infrastructure-as-code with auto-scaling, load balancing, and high availability.

**Step 1: Update Terraform variables**

Edit `infrastructure/terraform/variables.tf`:

```hcl
variable "environment" {
  default = "prod"
}

variable "min_instances" {
  default = 2  # High availability
}

variable "max_instances" {
  default = 10
}

variable "desired_instances" {
  default = 2
}
```

**Step 2: Get AMI ID**

```bash
# Get latest Deep Learning AMI for your region
aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*" \
              "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
    --output text

# Copy the AMI ID (e.g., ami-0123456789abcdef0)
```

Update `variables.tf`:
```hcl
variable "ami_id" {
  default = "ami-0123456789abcdef0"  # Your AMI ID
}
```

**Step 3: Deploy infrastructure**

```bash
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Deploy
terraform apply
```

**Expected time:** 15-20 minutes

**Terraform will create:**
- VPC with 2 public subnets
- Internet Gateway
- Security groups (ALB, EC2)
- IAM roles and policies
- DynamoDB tables (sessions, metrics)
- Application Load Balancer
- Launch template
- Auto Scaling Group (2-10 instances)
- CloudWatch alarms
- CloudWatch log group

**Step 4: Get endpoints**

```bash
terraform output

# Output:
# alb_dns_name = "socialsensear-alb-1234567890.us-east-1.elb.amazonaws.com"
# api_endpoint = "http://socialsensear-alb-1234567890.us-east-1.elb.amazonaws.com/api/v1"
# websocket_endpoint = "ws://socialsensear-alb-1234567890.us-east-1.elb.amazonaws.com/ws"
```

**Step 5: Test production deployment**

```bash
# Health check
curl http://YOUR_ALB_DNS/health

# Get status
curl -H "X-API-Key: YOUR_API_KEY" http://YOUR_ALB_DNS/api/v1/status
```

**Step 6: Connect clients**

Update your AR headset client configuration:

```python
# Production endpoint
server_url = "ws://YOUR_ALB_DNS"
api_key = "YOUR_PRODUCTION_API_KEY"
```

---

## Configuration

### Environment Variables

Configure the server using environment variables or `.env` file:

```bash
# Server settings
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Video settings
VIDEO_WIDTH=1280
VIDEO_HEIGHT=720
VIDEO_FPS=30
VIDEO_BITRATE=5000000

# Session settings
SESSION_TIMEOUT_SECONDS=300
SESSION_MAX_DURATION_SECONDS=14400

# AWS settings
AWS_REGION=us-east-1
DYNAMODB_TABLE_SESSIONS=socialsensear-sessions
DYNAMODB_TABLE_METRICS=socialsensear-metrics

# CloudWatch
CLOUDWATCH_NAMESPACE=SocialSenseAR
ENABLE_CLOUDWATCH=true

# Security
ALLOWED_ORIGINS=*  # Restrict in production
JWT_SECRET_KEY=your_secret_key_here
```

### Secrets Management

API keys and sensitive configuration are stored in AWS Secrets Manager:

```bash
# View secrets
aws secretsmanager get-secret-value \
    --secret-id socialsensear-secrets \
    --region us-east-1

# Update secrets
aws secretsmanager update-secret \
    --secret-id socialsensear-secrets \
    --secret-string '{"GEMINI_API_KEY":"new_key","OPENAI_API_KEY":"new_key"}' \
    --region us-east-1
```

---

## Monitoring

### CloudWatch Dashboards

**Create a dashboard:**

```bash
aws cloudwatch put-dashboard \
    --dashboard-name SocialSenseAR \
    --dashboard-body file://monitoring/dashboard.json
```

**Key metrics to monitor:**
- `SocialSenseAR/FrameLatency` - Processing latency (ms)
- `SocialSenseAR/ActiveSessions` - Concurrent users
- `SocialSenseAR/GPUUtilization` - GPU usage (%)
- `SocialSenseAR/ErrorRate` - Error percentage
- `AWS/EC2/CPUUtilization` - CPU usage

### CloudWatch Alarms

**High latency alarm:**
```bash
aws cloudwatch put-metric-alarm \
    --alarm-name socialsensear-high-latency \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --metric-name FrameLatency \
    --namespace SocialSenseAR \
    --period 300 \
    --statistic Average \
    --threshold 100 \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:alerts
```

### Logs

**View logs:**
```bash
# From CloudWatch
aws logs tail /socialsensear/server --follow

# From EC2 instance
ssh -i socialsensear-key.pem ubuntu@YOUR_INSTANCE_IP
docker logs -f socialsensear
```

---

## Scaling

### Auto-Scaling Configuration

**Scaling policies are based on:**

1. **CPU Utilization**
   - Scale OUT: CPU > 70% for 2 minutes → add 1 instance
   - Scale IN: CPU < 30% for 10 minutes → remove 1 instance

2. **Custom Metrics**
   - Scale based on active sessions
   - Scale based on GPU utilization
   - Scale based on average latency

**Modify scaling thresholds:**

Edit `infrastructure/terraform/main.tf`:

```hcl
resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  threshold = 70  # Change this
}
```

Then:
```bash
cd infrastructure/terraform
terraform apply
```

### Manual Scaling

**Temporarily adjust instance count:**

```bash
# Increase to 5 instances
aws autoscaling set-desired-capacity \
    --auto-scaling-group-name socialsensear-asg \
    --desired-capacity 5

# Decrease to 2 instances
aws autoscaling set-desired-capacity \
    --auto-scaling-group-name socialsensear-asg \
    --desired-capacity 2
```

---

## Cost Optimization

### 1. Use Spot Instances

**Savings:** 60-70% vs On-Demand

Edit `infrastructure/terraform/main.tf`:

```hcl
resource "aws_launch_template" "main" {
  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price = "0.30"  # Adjust based on region
    }
  }
}
```

**Trade-off:** AWS can terminate with 2-minute notice

### 2. Stop Instances When Not in Use

```bash
# Stop all instances in ASG (development only)
aws autoscaling set-desired-capacity \
    --auto-scaling-group-name socialsensear-asg \
    --desired-capacity 0

# Resume
aws autoscaling set-desired-capacity \
    --auto-scaling-group-name socialsensear-asg \
    --desired-capacity 2
```

### 3. Use Scheduled Scaling

**Scale down during off-hours:**

```bash
# Scale down at midnight
aws autoscaling put-scheduled-action \
    --auto-scaling-group-name socialsensear-asg \
    --scheduled-action-name scale-down-night \
    --recurrence "0 0 * * *" \
    --desired-capacity 1

# Scale up at 8am
aws autoscaling put-scheduled-action \
    --auto-scaling-group-name socialsensear-asg \
    --scheduled-action-name scale-up-morning \
    --recurrence "0 8 * * *" \
    --desired-capacity 2
```

### 4. Monitor Costs

**Set up billing alerts:**

```bash
aws cloudwatch put-metric-alarm \
    --alarm-name high-billing \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --metric-name EstimatedCharges \
    --namespace AWS/Billing \
    --period 86400 \
    --statistic Maximum \
    --threshold 500 \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:billing-alerts
```

---

## Troubleshooting

### Issue: Instances Not Starting

**Check:**
1. GPU instance limit quota
2. Security group rules
3. IAM role permissions

**Solution:**
```bash
# Check instance state
aws autoscaling describe-auto-scaling-groups \
    --auto-scaling-group-names socialsensear-asg

# Check recent activity
aws autoscaling describe-scaling-activities \
    --auto-scaling-group-name socialsensear-asg \
    --max-records 10
```

### Issue: High Latency

**Check:**
1. GPU utilization (should be < 90%)
2. Network bandwidth
3. Instance count (may need to scale out)

**Solution:**
```bash
# Get metrics
curl -H "X-API-Key: YOUR_KEY" http://YOUR_ALB_DNS/api/v1/metrics

# Manually scale out
aws autoscaling set-desired-capacity \
    --auto-scaling-group-name socialsensear-asg \
    --desired-capacity 5
```

### Issue: Load Balancer 502 Errors

**Check:**
1. Target group health
2. Instance health checks
3. Security group rules

**Solution:**
```bash
# Check target health
aws elbv2 describe-target-health \
    --target-group-arn YOUR_TARGET_GROUP_ARN

# Check security groups allow port 8000 from ALB
```

---

## Cleanup

### Remove Development Instance

```bash
# Get instance ID from instance_info.txt
INSTANCE_ID="i-0123456789abcdef0"

# Terminate
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

# Delete security group (wait until instance terminates)
aws ec2 delete-security-group --group-name socialsensear-sg

# Delete key pair
aws ec2 delete-key-pair --key-name socialsensear-key
rm socialsensear-key.pem
```

### Remove Production Infrastructure

```bash
cd infrastructure/terraform

# Destroy all resources
terraform destroy

# Confirm with: yes
```

**This will delete:**
- All EC2 instances
- Load balancer
- DynamoDB tables
- Security groups
- IAM roles
- CloudWatch log groups

**Note:** ECR repository and Secrets Manager entries must be deleted manually:

```bash
# Delete ECR
aws ecr delete-repository --repository-name socialsensear --force

# Delete secrets
aws secretsmanager delete-secret \
    --secret-id socialsensear-secrets \
    --force-delete-without-recovery
```

---

## Production Checklist

Before going to production:

- [ ] Update AMI ID for your region
- [ ] Set production API keys
- [ ] Enable HTTPS on ALB (add SSL certificate)
- [ ] Restrict security group SSH access to your IP
- [ ] Set up CloudWatch alarms with SNS notifications
- [ ] Configure auto-scaling thresholds
- [ ] Set up scheduled scaling for off-hours
- [ ] Enable CloudWatch detailed monitoring
- [ ] Set up billing alerts
- [ ] Test failover scenarios
- [ ] Document disaster recovery procedures
- [ ] Configure DynamoDB backups
- [ ] Set up S3 for log archival
- [ ] Implement CI/CD for automated deployments

---

## Support

For deployment issues:
- Check CloudWatch logs: `/socialsensear/server`
- Review EC2 instance system logs
- Check ALB access logs
- Monitor CloudWatch metrics dashboard
- Verify security group rules
- Test from EC2 instance directly to isolate ALB issues
