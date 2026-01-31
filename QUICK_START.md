# AWS Deployment Quick Start

**Goal**: Move your compute to AWS GPU instance for 5-10x faster performance.

## ⚡ 3-Step Deployment

### 1️⃣ Setup AWS (One-Time)

```bash
# Install AWS CLI (if not installed)
brew install awscli

# Configure credentials
aws configure
# Enter: Access Key ID, Secret Key, region (us-east-1), format (json)

# Make scripts executable
chmod +x *.sh
```

### 2️⃣ Deploy to AWS (~15 minutes)

```bash
# Build and push Docker image
./aws_deploy.sh

# Create GPU instance
./aws_ec2_setup.sh
```

**Save the output!** You'll get:
- SSH key file: `socialsensear-key.pem`
- Public IP address
- Instance ID

### 3️⃣ Start Application

```bash
# Connect to your instance (replace with your IP)
ssh -i socialsensear-key.pem ubuntu@YOUR_PUBLIC_IP

# Copy run script to instance (from your computer)
scp -i socialsensear-key.pem aws_run_container.sh ubuntu@YOUR_PUBLIC_IP:~/

# On the instance, run:
chmod +x aws_run_container.sh
./aws_run_container.sh
```

**Done!** Your app is running on AWS GPU at 30-60 FPS.

---

## 💰 Cost Control

**Save money when not using:**
```bash
# Stop instance (your computer, not SSH)
aws ec2 stop-instances --instance-ids YOUR_INSTANCE_ID --region us-east-1

# Start when needed
aws ec2 start-instances --instance-ids YOUR_INSTANCE_ID --region us-east-1
```

**Costs:**
- Running: ~$0.15/hour (Spot) or $0.53/hour (On-Demand)
- Stopped: ~$0.15/day (storage only)
- Monthly (8hr/day): ~$36/month

---

## 🔍 Verify It's Working

```bash
# Check GPU (on instance)
nvidia-smi

# View logs
docker logs -f socialsensear

# Look for these in logs:
# ✓ "CUDA available: True"
# ✓ "GPU: NVIDIA Tesla T4"
# ✓ Processing at 30+ FPS
```

---

## 📚 Full Documentation

See `AWS_SETUP_GUIDE.md` for:
- Detailed troubleshooting
- Performance optimization
- Security best practices
- Cost optimization tips

---

## 🆘 Quick Troubleshooting

**Can't SSH?**
- Wait 3-5 minutes after instance creation
- Check IP didn't change (happens after stop/start)

**GPU not working?**
- Run `nvidia-smi` - should show NVIDIA T4
- Check logs: `docker logs socialsensear`

**High costs?**
- Stop instance when not using
- Use Spot instances (set in aws_ec2_setup.sh)

---

## 📝 What You Need

**One-time info to save:**
- Instance ID (from `instance_info.txt`)
- Public IP (changes when you stop/start)
- SSH key file (`socialsensear-key.pem`)

**AWS Credentials:**
- Get from AWS Console → IAM → Security Credentials
- Create new access key if needed

---

That's it! Full guide in `AWS_SETUP_GUIDE.md`.
