# Deployment Guide - RAG Chatbot

This guide covers various deployment options for the RAG Chatbot application.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development Deployment](#local-development-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
  - [AWS Deployment](#aws-deployment)
  - [Google Cloud Platform](#google-cloud-platform)
  - [Azure Deployment](#azure-deployment)
- [Environment Variables](#environment-variables)
- [Production Considerations](#production-considerations)
- [Monitoring and Logging](#monitoring-and-logging)
- [Scaling](#scaling)
- [Security](#security)

---

## Prerequisites

Before deploying, ensure you have:

- ✅ Python 3.8+ installed
- ✅ Valid API key (OpenAI or Google)
- ✅ Sufficient storage for document uploads and vector database
- ✅ Network access for API calls
- ✅ (For cloud) Cloud provider account and credentials

---

## Local Development Deployment

### Standard Installation

1. **Clone/Download the project**
   ```bash
   cd "capstone project 1"
   ```

2. **Create virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   Copy-Item .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   ```
   Open browser: http://127.0.0.1:7860
   ```

### Running in Background (Windows)

```powershell
# Using Start-Process
Start-Process python -ArgumentList "app.py" -WindowStyle Hidden

# Or using nohup equivalent with PowerShell
powershell -Command "python app.py > app.log 2>&1" -WindowStyle Hidden
```

### Running in Background (Linux/Mac)

```bash
nohup python app.py > app.log 2>&1 &
```

---

## Docker Deployment

### Create Dockerfile

Create `Dockerfile` in the project root:

```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data chroma_db

# Expose port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]
```

### Create docker-compose.yml

```yaml
version: '3.8'

services:
  rag-chatbot:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./data:/app/data
      - ./chroma_db:/app/chroma_db
    environment:
      - LLM_PROVIDER=${LLM_PROVIDER}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=${OPENAI_MODEL}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - GOOGLE_MODEL=${GOOGLE_MODEL}
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker build -t rag-chatbot .

# Run container
docker run -p 7860:7860 --env-file .env rag-chatbot

# Or use docker-compose
docker-compose up -d
```

### Docker Commands

```bash
# View logs
docker logs -f rag-chatbot

# Stop container
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

---

## Cloud Deployment

### AWS Deployment

#### Option 1: AWS EC2

1. **Launch EC2 Instance**
   - AMI: Ubuntu Server 22.04 LTS
   - Instance Type: t3.medium or larger
   - Storage: 20GB+ EBS
   - Security Group: Allow port 7860

2. **Connect to instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Install dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv -y
   ```

4. **Deploy application**
   ```bash
   git clone <your-repo>
   cd "capstone project 1"
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Copy .env file with API keys
   nano .env
   
   # Run with nohup
   nohup python app.py > app.log 2>&1 &
   ```

5. **Access application**
   ```
   http://your-ec2-public-ip:7860
   ```

#### Option 2: AWS ECS (Fargate)

1. **Create ECR repository**
   ```bash
   aws ecr create-repository --repository-name rag-chatbot
   ```

2. **Build and push Docker image**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <your-account>.dkr.ecr.us-east-1.amazonaws.com
   
   docker build -t rag-chatbot .
   docker tag rag-chatbot:latest <your-account>.dkr.ecr.us-east-1.amazonaws.com/rag-chatbot:latest
   docker push <your-account>.dkr.ecr.us-east-1.amazonaws.com/rag-chatbot:latest
   ```

3. **Create ECS task definition and service** (via AWS Console or CLI)

4. **Configure environment variables in ECS task**

#### Option 3: AWS Elastic Beanstalk

1. **Install EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize and deploy**
   ```bash
   eb init -p python-3.10 rag-chatbot
   eb create rag-chatbot-env
   eb setenv OPENAI_API_KEY=your-key
   eb deploy
   ```

### Google Cloud Platform

#### Option 1: Google Compute Engine

Similar to AWS EC2, create a VM instance and deploy manually.

#### Option 2: Google Cloud Run

1. **Build and push to Container Registry**
   ```bash
   gcloud builds submit --tag gcr.io/your-project/rag-chatbot
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy rag-chatbot \
     --image gcr.io/your-project/rag-chatbot \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars OPENAI_API_KEY=your-key
   ```

### Azure Deployment

#### Option 1: Azure Virtual Machine

Similar to AWS EC2, create a VM and deploy manually.

#### Option 2: Azure Container Instances

1. **Create container registry**
   ```bash
   az acr create --resource-group myResourceGroup --name myregistry --sku Basic
   ```

2. **Build and push**
   ```bash
   az acr build --registry myregistry --image rag-chatbot .
   ```

3. **Deploy container**
   ```bash
   az container create \
     --resource-group myResourceGroup \
     --name rag-chatbot \
     --image myregistry.azurecr.io/rag-chatbot \
     --dns-name-label rag-chatbot \
     --ports 7860 \
     --environment-variables OPENAI_API_KEY=your-key
   ```

---

## Environment Variables

### Required Variables

```env
# LLM Configuration (Choose one)
LLM_PROVIDER=openai              # or "google"

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo

# Google
GOOGLE_API_KEY=AIza...
GOOGLE_MODEL=gemini-pro
```

### Optional Variables

```env
# Gradio Configuration
GRADIO_SHARE=False               # Set True for public URL
GRADIO_SERVER_PORT=7860
GRADIO_SERVER_NAME=0.0.0.0      # For cloud deployment

# Application Settings
CHUNK_SIZE_MIN=500
CHUNK_SIZE_MAX=800
CHUNK_OVERLAP_PERCENT=10
TOP_K_RETRIEVAL=5
MAX_TOKENS=1000
TEMPERATURE=0.7
```

---

## Production Considerations

### 1. Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Gradio
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 2. SSL/TLS with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 3. Process Manager (systemd)

Create `/etc/systemd/system/rag-chatbot.service`:

```ini
[Unit]
Description=RAG Chatbot Application
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/capstone project 1
Environment="PATH=/home/ubuntu/capstone project 1/venv/bin"
ExecStart=/home/ubuntu/capstone project 1/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable rag-chatbot
sudo systemctl start rag-chatbot
sudo systemctl status rag-chatbot
```

### 4. Database Persistence

Ensure `chroma_db` directory is persisted:
- Use mounted volumes in Docker
- Use persistent storage in cloud (EBS, Persistent Disk)
- Regular backups

### 5. File Upload Limits

Modify `src/config.py`:
```python
MAX_FILE_SIZE_MB = 50  # Adjust as needed
```

---

## Monitoring and Logging

### Application Logs

```python
# Already configured in the application
import logging
logging.basicConfig(level=logging.INFO)
```

### View Logs

```bash
# Docker
docker logs -f rag-chatbot

# Systemd
sudo journalctl -u rag-chatbot -f

# File-based
tail -f app.log
```

### Cloud Monitoring

- **AWS**: CloudWatch Logs
- **GCP**: Cloud Logging
- **Azure**: Application Insights

---

## Scaling

### Horizontal Scaling

1. **Load Balancer**: Distribute traffic across multiple instances
2. **Shared Vector Store**: Use a shared database (consider Pinecone, Weaviate)
3. **Session Management**: Implement Redis for session state

### Vertical Scaling

- Increase instance size (CPU, RAM)
- Optimize embedding batch size
- Cache frequently accessed chunks

### Performance Optimization

```python
# Adjust in config.py
TOP_K_RETRIEVAL = 3  # Reduce for faster response
CHUNK_SIZE_MAX = 600  # Smaller chunks = more granular search
```

---

## Security

### 1. API Key Security

- ✅ Never commit `.env` to version control
- ✅ Use environment variables
- ✅ Rotate keys regularly
- ✅ Use secrets management (AWS Secrets Manager, GCP Secret Manager)

### 2. Network Security

- ✅ Use HTTPS in production
- ✅ Implement rate limiting
- ✅ Configure firewall rules
- ✅ Use VPC/Private networks

### 3. Input Validation

Already implemented in the application:
- File type validation
- File size limits
- Text sanitization

### 4. Authentication (Optional)

Add authentication to Gradio:

```python
interface.launch(
    auth=("username", "password"),
    # or
    auth_message="Enter credentials"
)
```

---

## Health Checks

### Basic Health Check Endpoint

Add to `app.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

app_fastapi = FastAPI()

@app_fastapi.get("/health")
def health_check():
    return {"status": "healthy"}

# Mount Gradio app
app_fastapi.mount("/", WSGIMiddleware(interface))
```

---

## Backup and Recovery

### Vector Database Backup

```bash
# Backup chroma_db directory
tar -czf chroma_db_backup_$(date +%Y%m%d).tar.gz chroma_db/

# Restore
tar -xzf chroma_db_backup_20240101.tar.gz
```

### Automated Backups

```bash
# Add to crontab (Linux)
0 2 * * * cd /path/to/project && tar -czf backups/chroma_db_$(date +\%Y\%m\%d).tar.gz chroma_db/
```

---

## Cost Optimization

### LLM API Costs

- Use `gpt-3.5-turbo` instead of `gpt-4` (lower cost)
- Cache common queries
- Limit `MAX_TOKENS` to reduce costs
- Monitor usage with API dashboards

### Cloud Costs

- Use auto-scaling to match demand
- Reserve instances for steady workloads
- Clean up unused resources
- Monitor with cost alerts

---

## Troubleshooting

### Common Production Issues

**1. Memory Issues**
```bash
# Increase instance memory or reduce batch sizes
# Monitor with: free -h
```

**2. Port Conflicts**
```bash
# Check port usage: netstat -tlnp | grep 7860
# Change port in config
```

**3. Permission Issues**
```bash
# Ensure correct ownership
sudo chown -R ubuntu:ubuntu /path/to/project
```

**4. API Rate Limits**
```
# Implement retry logic and exponential backoff
# Monitor API usage dashboards
```

---

## Support and Maintenance

### Regular Maintenance Tasks

- ✅ Update dependencies: `pip install -U -r requirements.txt`
- ✅ Review and rotate API keys
- ✅ Monitor disk space: `df -h`
- ✅ Review application logs
- ✅ Backup vector database
- ✅ Update security patches

### Monitoring Checklist

- [ ] Application is responding
- [ ] API keys are valid
- [ ] Disk space available
- [ ] Memory usage normal
- [ ] No error spikes in logs
- [ ] Backup completed

---

## Conclusion

This deployment guide covers the main deployment scenarios. Choose the option that best fits your requirements:

- **Local Development**: Quick testing and development
- **Docker**: Consistent environments and easy deployment
- **Cloud**: Scalable, production-ready deployment

For questions or issues, refer to the main [README.md](README.md) or open an issue.

---

**Last Updated**: November 2025
