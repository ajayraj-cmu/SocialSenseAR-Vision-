# SocialSenseAR AWS Deployment Guide

Complete guide to deploying your real-time vision application on AWS GPU instances for **5-10x performance improvement**.

## 🎯 What This Does

Moves your compute-intensive vision processing (FastSAM + YOLO-World) from your local CPU to AWS GPU instances:
- **Before**: CPU-only, ~10-15 FPS
- **After**: GPU-accelerated (NVIDIA T4), 30-60+ FPS
- **Your computer**: Just renders the output, runs smoothly

---

## 💰 Cost Estimate

### Recommended: g4dn.xlarge Spot Instance
- **GPU**: NVIDIA T4 (16GB VRAM)
- **Compute**: 4 vCPUs, 16GB RAM
- **Cost**: ~$0.15/hour (Spot) or $0.526/hour (On-Demand)
- **Monthly** (24/7): ~$110/month (Spot) or $380/month (On-Demand)

### Budget Alternative: Stop when not in use
- Run only 8 hours/day: ~$36/month (Spot)
- Start/stop as needed: Pay per hour

### Additional Costs
- **Storage**: ~$5-10/month (EBS volumes)
- **APIs**: $20-100/month (Gemini, OpenAI - based on usage)
- **Total**: ~$60-150/month for typical usage

---

## 📋 Prerequisites

### On Your Computer (One-Time Setup)

1. **AWS Account**
   - Sign up at https://aws.amazon.com
   - Add payment method
   - Request g4dn instance limit increase if needed (Service Quotas)

2. **Install AWS CLI**
   ```bash
   # macOS
   brew install awscli

   # Or download from: https://aws.amazon.com/cli/
   ```

3. **Install Docker Desktop**
   - Download from: https://www.docker.com/products/docker-desktop
   - Install and start Docker

4. **Configure AWS Credentials**
   ```bash
   aws configure
   ```
   Enter:
   - AWS Access Key ID (from AWS Console → IAM → Security Credentials)
   - AWS Secret Access Key
   - Default region: `us-east-1` (or your preferred region)
   - Default output format: `json`

5. **Make Scripts Executable**
   ```bash
   chmod +x aws_deploy.sh aws_ec2_setup.sh aws_run_container.sh
   ```

---

## 🚀 Deployment Steps

### Step 1: Build and Push Docker Image (On Your Computer)

This uploads your application to AWS's container registry.

```bash
./aws_deploy.sh
```

**What this does:**
- Creates AWS ECR (container registry)
- Builds Docker image with GPU support
- Pushes image to AWS
- Uploads your API keys to AWS Secrets Manager (secure storage)

**Expected time**: 10-15 minutes (first time)

**Output**: You'll see ✓ checkmarks for each step

---

### Step 2: Create GPU Instance (On Your Computer)

This creates your AWS server with GPU.

```bash
./aws_ec2_setup.sh
```

**What this does:**
- Creates SSH key pair (saved as `socialsensear-key.pem`)
- Sets up security group (firewall rules)
- Creates IAM role (permissions)
- Launches g4dn.xlarge GPU instance
- Installs Docker + NVIDIA drivers

**Expected time**: 5-10 minutes

**Output**: You'll get:
- Instance ID (e.g., `i-0123456789abcdef0`)
- Public IP address (e.g., `54.123.45.67`)
- SSH command to connect

**Save these!** They're also written to `instance_info.txt`

---

### Step 3: Connect to Your Instance

Wait 2-3 minutes after Step 2, then connect:

```bash
ssh -i socialsensear-key.pem ubuntu@<YOUR_PUBLIC_IP>
```

Replace `<YOUR_PUBLIC_IP>` with the IP from Step 2 output.

**First-time SSH**: Type `yes` when asked about fingerprint

---

### Step 4: Start Your Application (On AWS Instance)

Once connected via SSH, run:

```bash
# Option A: Create the run script manually
cat > aws_run_container.sh << 'EOF'
[paste contents of aws_run_container.sh]
EOF
chmod +x aws_run_container.sh

# Option B: Copy from your computer (easier)
# Exit SSH (type 'exit'), then on your computer:
scp -i socialsensear-key.pem aws_run_container.sh ubuntu@<YOUR_PUBLIC_IP>:~/

# Reconnect via SSH
ssh -i socialsensear-key.pem ubuntu@<YOUR_PUBLIC_IP>
chmod +x aws_run_container.sh
```

Then run:

```bash
./aws_run_container.sh
```

**What this does:**
- Verifies GPU is working
- Downloads your Docker image from ECR
- Retrieves API keys from Secrets Manager
- Starts your application with GPU acceleration

**Expected time**: 3-5 minutes

**Success indicators:**
- You see `✓ Container started`
- Logs show: `CUDA available: True`
- GPU is detected in nvidia-smi output

---

## 🔍 Monitoring & Management

### Check Application Status

```bash
# View live logs
docker logs -f socialsensear

# Check if running
docker ps

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Access Output Files

Output files are saved to the instance at:
- Recordings: `~/socialsensear-data/`
- Logs: `~/socialsensear-logs/`

To download to your computer:
```bash
# On your computer (not SSH)
scp -i socialsensear-key.pem -r ubuntu@<YOUR_PUBLIC_IP>:~/socialsensear-data/ ./
```

---

## 💾 Start/Stop Instance (Save Money!)

### Stop Instance When Not Using
```bash
# On your computer (not SSH)
aws ec2 stop-instances --instance-ids <INSTANCE_ID> --region us-east-1
```
**Saves money!** You only pay for storage (~$5/month) when stopped.

### Start Instance When Needed
```bash
aws ec2 start-instances --instance-ids <INSTANCE_ID> --region us-east-1

# Get new public IP (it changes after stop/start)
aws ec2 describe-instances --instance-ids <INSTANCE_ID> \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text --region us-east-1
```

Then reconnect and restart container:
```bash
ssh -i socialsensear-key.pem ubuntu@<NEW_PUBLIC_IP>
./aws_run_container.sh
```

---

## 🔧 Troubleshooting

### Container Won't Start

**Check logs:**
```bash
docker logs socialsensear
```

**Common issues:**
1. **GPU not detected**: Run `nvidia-smi` - if error, instance may need restart
2. **API keys missing**: Check AWS Secrets Manager has `socialsensear-secrets`
3. **Out of memory**: Reduce model batch size in config

### Can't Connect via SSH

1. **Wait longer**: Instance may still be initializing (wait 5 minutes)
2. **Check security group**: Port 22 should be open
3. **Wrong IP**: IP changes after stop/start - get latest with:
   ```bash
   aws ec2 describe-instances --instance-ids <INSTANCE_ID> \
       --query 'Reservations[0].Instances[0].PublicIpAddress' \
       --output text --region us-east-1
   ```

### Slow Performance

1. **Check GPU usage**: `nvidia-smi` should show activity
2. **Verify GPU in logs**: Look for `CUDA available: True`
3. **Instance type**: Confirm you're using g4dn.xlarge, not CPU instance

### High AWS Bills

1. **Stop instance when not using**
2. **Use Spot instances** (edit `INSTANCE_TYPE` in script)
3. **Monitor with AWS Cost Explorer**
4. **Set billing alerts** in AWS Console

---

## 🔒 Security Best Practices

### Rotate API Keys

Your Gemini API key is currently exposed in the git repo. Rotate it:

1. Get new key from Google AI Studio
2. Update `.env` file locally
3. Re-run `./aws_deploy.sh` to upload new secrets

### Secure SSH Access

Restrict SSH to your IP only:
```bash
# Get your IP
MY_IP=$(curl -s ifconfig.me)

# Update security group
aws ec2 authorize-security-group-ingress \
    --group-name socialsensear-sg \
    --protocol tcp --port 22 \
    --cidr ${MY_IP}/32 \
    --region us-east-1

# Remove old wide-open rule
aws ec2 revoke-security-group-ingress \
    --group-name socialsensear-sg \
    --protocol tcp --port 22 \
    --cidr 0.0.0.0/0 \
    --region us-east-1
```

---

## 📊 Performance Optimization

### Enable GPU Acceleration

The Dockerfile automatically configures GPU support. Verify in logs:
```
CUDA available: True
GPU: NVIDIA Tesla T4
```

### Reduce Latency Further

Edit `config/settings.yaml` on the instance:
```yaml
segmentation_interval_frames: 1  # Process every frame (was 3)
downscale_factor: 0.75           # Higher resolution (was 0.5)
```

Restart container:
```bash
docker restart socialsensear
```

### Use Spot Instances (70% Cost Savings)

Edit `aws_ec2_setup.sh` and replace the `run-instances` command with:
```bash
aws ec2 run-instances \
    --instance-market-options '{"MarketType":"spot"}' \
    # ... rest of command
```

**Trade-off**: AWS can terminate with 2-minute notice if capacity needed

---

## 🗑️ Cleanup / Termination

### Terminate Everything (Delete All Resources)

```bash
# Get instance ID from instance_info.txt
INSTANCE_ID="<your-instance-id>"

# Terminate instance
aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region us-east-1

# Delete ECR repository
aws ecr delete-repository --repository-name socialsensear --force --region us-east-1

# Delete secrets
aws secretsmanager delete-secret --secret-id socialsensear-secrets --force-delete-without-recovery --region us-east-1

# Delete security group (wait until instance terminates)
aws ec2 delete-security-group --group-name socialsensear-sg --region us-east-1

# Delete IAM resources
aws iam remove-role-from-instance-profile --instance-profile-name socialsensear-instance-profile --role-name socialsensear-ec2-role
aws iam delete-instance-profile --instance-profile-name socialsensear-instance-profile
aws iam detach-role-policy --role-name socialsensear-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam detach-role-policy --role-name socialsensear-ec2-role --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
aws iam detach-role-policy --role-name socialsensear-ec2-role --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
aws iam delete-role --role-name socialsensear-ec2-role

# Delete key pair
aws ec2 delete-key-pair --key-name socialsensear-key --region us-east-1
rm socialsensear-key.pem
```

---

## 📞 Getting Help

### AWS Documentation
- EC2 GPU Instances: https://aws.amazon.com/ec2/instance-types/g4/
- ECR Guide: https://docs.aws.amazon.com/ecr/
- Secrets Manager: https://docs.aws.amazon.com/secretsmanager/

### Check Service Status
- Instance state: `aws ec2 describe-instances --instance-ids <INSTANCE_ID>`
- Container logs: `docker logs socialsensear`
- GPU status: `nvidia-smi`

### Common Commands Reference

```bash
# Local (your computer)
./aws_deploy.sh              # Build and push to AWS
./aws_ec2_setup.sh           # Create instance
aws ec2 stop-instances       # Stop instance
aws ec2 start-instances      # Start instance

# SSH into instance
ssh -i socialsensear-key.pem ubuntu@<IP>

# On AWS instance
./aws_run_container.sh       # Start application
docker logs -f socialsensear # View logs
docker restart socialsensear # Restart app
nvidia-smi                   # Check GPU
htop                         # Check CPU/RAM
```

---

## 🎉 Success Checklist

- [ ] AWS account created and configured
- [ ] Docker image built and pushed to ECR
- [ ] EC2 GPU instance running
- [ ] SSH connection working
- [ ] Container started successfully
- [ ] Logs show "CUDA available: True"
- [ ] nvidia-smi shows GPU usage
- [ ] Application processing video at 30+ FPS
- [ ] API keys working (Gemini, OpenAI)
- [ ] Can stop/start instance to save money

---

## 🚀 Next Steps

1. **Test your application**: Verify performance meets expectations
2. **Set up monitoring**: CloudWatch dashboards for GPU utilization
3. **Configure auto-scaling**: For production workloads
4. **Add load balancer**: If exposing via web interface
5. **Set up CI/CD**: Auto-deploy on code changes

**Questions?** Check `instance_info.txt` for your specific instance details.
