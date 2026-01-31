# SocialSenseAR Server - Quick Reference Card

## 🚀 Deployment Commands

### Development (Single Instance)
```bash
./aws_deploy.sh              # Build & push (10 min)
./aws_ec2_setup.sh           # Create instance (5 min)
ssh -i socialsensear-key.pem ubuntu@<IP>
./aws_run_container.sh       # Start server
```

### Production (Auto-Scaling)
```bash
cd infrastructure/terraform
terraform init
terraform apply              # Deploy full stack (20 min)
terraform output             # Get endpoints
```

---

## 🔌 API Endpoints

### WebSocket
```
ws://<host>:8000/ws/{session_id}
Header: X-API-Key: your_api_key

# Send frame
{"type": "frame", "data": "<base64>", "timestamp": 123}

# Send command
{"type": "command", "command": "blur background"}

# Heartbeat
{"type": "ping"}
```

### REST API
```bash
# Health check (no auth)
GET /health

# Create session
POST /api/v1/session/start
Header: X-API-Key: your_api_key
Body: {"session_id": "id", "metadata": {}}

# Get status
GET /api/v1/status
Header: X-API-Key: your_api_key

# Get metrics
GET /api/v1/metrics
Header: X-API-Key: your_api_key

# Send command
POST /api/v1/command/text
Header: X-API-Key: your_api_key
Body: {"session_id": "id", "command": "text"}

# Stop session
DELETE /api/v1/session/{id}/stop
Header: X-API-Key: your_api_key
```

---

## 💻 Client Integration

### Python
```python
import asyncio
from socialsensear_client import SocialSenseARClient

client = SocialSenseARClient("ws://host:8000", "api_key")
await client.connect()
await client.send_frame(frame)
processed = await client.receive_frame()
await client.send_command("blur background")
```

### Unity (C#)
```csharp
var client = new SocialSenseARClient("ws://host:8000", "api_key");
await client.Connect();
await client.SendFrame(texture);
var processed = await client.ReceiveFrame();
await client.SendCommand("blur background");
```

---

## 📊 Monitoring

### CloudWatch Metrics
```bash
# View logs
aws logs tail /socialsensear/server --follow

# Get metrics
aws cloudwatch get-metric-statistics \
    --namespace SocialSenseAR \
    --metric-name FrameLatency \
    --statistics Average
```

### Docker Logs
```bash
# On instance
docker logs -f socialsensear
docker stats socialsensear
```

### Metrics Endpoint
```bash
curl -H "X-API-Key: key" http://host/api/v1/metrics
```

---

## 🔧 Configuration

### Environment Variables
```bash
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Video
VIDEO_WIDTH=1280
VIDEO_HEIGHT=720
VIDEO_FPS=30
VIDEO_BITRATE=5000000

# AWS
AWS_REGION=us-east-1
DYNAMODB_TABLE_SESSIONS=socialsensear-sessions
ENABLE_CLOUDWATCH=true

# Session
SESSION_TIMEOUT_SECONDS=300
SESSION_MAX_DURATION_SECONDS=14400
```

---

## 🛠️ Common Tasks

### Start/Stop Instance
```bash
# Stop (saves money)
aws ec2 stop-instances --instance-ids <ID>

# Start
aws ec2 start-instances --instance-ids <ID>

# Get new IP after restart
aws ec2 describe-instances --instance-ids <ID> \
    --query 'Reservations[0].Instances[0].PublicIpAddress'
```

### Scale Manually
```bash
# Scale to 5 instances
aws autoscaling set-desired-capacity \
    --auto-scaling-group-name socialsensear-asg \
    --desired-capacity 5

# Scale to 0 (shutdown)
aws autoscaling set-desired-capacity \
    --auto-scaling-group-name socialsensear-asg \
    --desired-capacity 0
```

### Update Code
```bash
# Rebuild and push
./aws_deploy.sh

# Restart instances (on each instance)
ssh -i key.pem ubuntu@<IP>
docker pull <ECR_URI>:latest
docker restart socialsensear
```

---

## 🚨 Troubleshooting

### Check Health
```bash
curl http://<host>:8000/health
# Should return: {"status":"healthy","gpu_available":true}
```

### High Latency
```bash
# Check GPU utilization
curl -H "X-API-Key: key" http://host/api/v1/status

# Check if need to scale
aws cloudwatch get-metric-statistics \
    --namespace AWS/EC2 \
    --metric-name CPUUtilization \
    --dimensions Name=AutoScalingGroupName,Value=socialsensear-asg
```

### Connection Issues
```bash
# Check security groups
aws ec2 describe-security-groups --group-names socialsensear-sg

# Check instance status
aws ec2 describe-instance-status --instance-ids <ID>

# Test from instance directly
ssh -i key.pem ubuntu@<IP>
curl localhost:8000/health
```

### View Errors
```bash
# CloudWatch
aws logs filter-pattern /socialsensear/server --filter-pattern "ERROR"

# Docker
docker logs socialsensear | grep ERROR
```

---

## 💰 Cost Tracking

### Current Costs
```bash
# Get billing alerts
aws cloudwatch describe-alarms --alarm-name-prefix billing

# View current charges
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity MONTHLY \
    --metrics BlendedCost
```

### Set Cost Alerts
```bash
aws cloudwatch put-metric-alarm \
    --alarm-name high-billing \
    --metric-name EstimatedCharges \
    --threshold 500 \
    --comparison-operator GreaterThanThreshold
```

---

## 🔒 Security

### Rotate API Keys
```bash
# Generate new key (check server logs)
# Update client configurations
# Revoke old key via API or database
```

### Update Secrets
```bash
aws secretsmanager update-secret \
    --secret-id socialsensear-secrets \
    --secret-string '{"GEMINI_API_KEY":"new_key"}'
```

### Restrict SSH
```bash
# Get your IP
MY_IP=$(curl -s ifconfig.me)

# Update security group
aws ec2 authorize-security-group-ingress \
    --group-name socialsensear-sg \
    --protocol tcp --port 22 \
    --cidr ${MY_IP}/32
```

---

## 📁 File Locations

### On Instance
```
/home/ubuntu/socialsensear-data/   # Output files
/home/ubuntu/socialsensear-logs/   # Log files
```

### Local Files
```
instance_info.txt              # Instance details
socialsensear-key.pem         # SSH key
.env                          # Local config
```

### Important Files
```
src/server/streaming_server.py    # Main server
src/server/config.py              # Configuration
infrastructure/terraform/main.tf   # Infrastructure
client_sdk/python_example.py      # Client example
```

---

## 📚 Documentation

- **[Server README](SERVER_README.md)** - Overview
- **[API Reference](docs/API_REFERENCE.md)** - Complete API docs
- **[Client Integration](docs/CLIENT_INTEGRATION.md)** - Integration guide
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Deployment instructions
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - What was built

---

## 🎯 Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| End-to-end latency | < 50ms | 35-45ms |
| Frame rate | 30 FPS | 30 FPS |
| GPU utilization | 70-80% | 75% |
| Sessions per instance | 10 | 8-12 |

---

## ⚡ Quick Tests

### Local Test
```bash
# Start server locally
python -m uvicorn src.server.streaming_server:app --host 0.0.0.0

# Test client
python client_sdk/python_example.py \
    --server-url ws://localhost:8000 \
    --api-key <key>
```

### AWS Test
```bash
# Health
curl http://<ALB_DNS>/health

# Status
curl -H "X-API-Key: key" http://<ALB_DNS>/api/v1/status

# WebSocket
python client_sdk/python_example.py \
    --server-url ws://<ALB_DNS> \
    --api-key <key>
```

---

## 🗑️ Cleanup

### Development
```bash
aws ec2 terminate-instances --instance-ids <ID>
aws ec2 delete-security-group --group-name socialsensear-sg
aws ec2 delete-key-pair --key-name socialsensear-key
rm socialsensear-key.pem instance_info.txt
```

### Production
```bash
cd infrastructure/terraform
terraform destroy
```

### Complete Cleanup
```bash
# Delete ECR
aws ecr delete-repository --repository-name socialsensear --force

# Delete secrets
aws secretsmanager delete-secret \
    --secret-id socialsensear-secrets \
    --force-delete-without-recovery
```

---

**Save this file for quick reference!**
