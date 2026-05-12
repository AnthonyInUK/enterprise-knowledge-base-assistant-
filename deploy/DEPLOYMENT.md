# Server Deployment (Ubuntu 22.04 + IP/Domain)

This repo can be deployed as a single-node service:
- FastAPI API (`/v1/*`)
- React frontend (static)
- PostgreSQL + pgvector
- background worker (async job queue)

## 1) Prepare server

Install Docker:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```

Log out/in once after adding the docker group.

## 2) DNS (recommended)

Point `anthonyaiagent.works` A record to your server IP.
Then you can use HTTPS automatically via Caddy.

## 3) Configure env

On server, in repo root:

```bash
cp deploy/.env.prod.example deploy/.env.prod
```

Edit `deploy/.env.prod`:
- `SITE_ADDRESS=anthonyaiagent.works` (HTTPS)
- If you want IP-only HTTP: `SITE_ADDRESS=:80`
- Configure LLM:
  - If using remote Ollama: set `OLLAMA_URL=http://<your-ollama-host>:11434`
  - If using Claude: set `RAG_LLM_BACKEND=claude` + `ANTHROPIC_API_KEY=...`

## 4) Start services

```bash
docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env.prod up -d --build
```

Check status:

```bash
docker compose -f deploy/docker-compose.prod.yml ps
docker compose -f deploy/docker-compose.prod.yml logs -n 100 --no-log-prefix api
```

Health:
- `http(s)://<SITE_ADDRESS>/` (frontend)
- `http(s)://<SITE_ADDRESS>/v1/health`

## 5) Updates

Pull latest code and rebuild:

```bash
git pull
docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env.prod up -d --build
```

