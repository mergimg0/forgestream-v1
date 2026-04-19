# RunPod CRQA Endpoint Deployment

## Prerequisites

- RunPod account with API key configured
- Docker installed locally
- Docker Hub account (or any container registry)

## SSH Key Setup (One-Time)

1. Go to https://www.runpod.io/console/user/settings
2. Under "SSH Public Keys", add your key:
   ```
   ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJy2n5H4CPE7RpTe7ZLqTKF8HH2E96zltnPD/zxuqxn9
   ```

## API Key

Stored at `.secrets/runpod_api_key.txt` (gitignored).

## Deployment Steps

### Option A: Serverless (Recommended — Scales to Zero)

```bash
# 1. Build and push Docker image
cd ~/projects/forgestream/runpod
docker build -t mergimg0/forgestream-crqa:latest .
docker push mergimg0/forgestream-crqa:latest

# 2. Create serverless endpoint via RunPod dashboard
#    - Go to https://www.runpod.io/console/serverless
#    - Create new endpoint
#    - Docker image: mergimg0/forgestream-crqa:latest
#    - Min workers: 0
#    - Max workers: 1
#    - GPU: Not required (CPU-only endpoint)
#    - Note the endpoint ID

# 3. Configure ForgeStream
export FORGESTREAM_RUNPOD_CRQA_ENDPOINT=https://api.runpod.ai/v2/<endpoint-id>
```

### Option B: Persistent Pod

```bash
# 1. Create pod via CLI
runpodctl create pod --name forgestream-crqa \
  --image runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 \
  --gpuType "NVIDIA RTX A4000" --gpuCount 1 \
  --ports 8000/http

# 2. SSH into pod and deploy
ssh <pod-id>@ssh.runpod.io
pip install fastapi uvicorn numpy
# Upload crqa_endpoint.py to /workspace/
nohup uvicorn crqa_endpoint:app --host 0.0.0.0 --port 8000 &

# 3. Configure ForgeStream
export FORGESTREAM_RUNPOD_CRQA_ENDPOINT=https://<pod-id>-8000.proxy.runpod.net
```

## Testing

```bash
# Health check
curl https://<endpoint>/health

# Test CRQA validation
curl -X POST https://<endpoint>/crqa/validate \
  -H "Content-Type: application/json" \
  -d '{"f0_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
       "f0_b": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
       "params": {"radius": 0.25, "n_surrogates": 5}}'
```
