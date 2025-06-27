# AI Coder Meta-Agent: New Project Setup Guide

This guide provides step-by-step instructions for new developers to set up a fresh software project managed by the AI Coder Meta-Agent. This setup creates a dedicated monorepo structure, integrates the AI Coder services, and initializes your new project for AI-driven development.

---

## 1. System Requirements

Before you begin, ensure your development machine (PC or Mac) has the following software installed:

- **Git:** Required for cloning the AI Coder Meta-Agent repository.  
  [Download Git](https://git-scm.com/downloads)

- **Docker Desktop:** Includes Docker Engine and Docker Compose.  
  [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)  
  Verify with:
  ```bash
  docker --version
  docker compose version
  ```

- **GitHub CLI (`gh`):** For automating repository creation and initial push.  
  [GitHub CLI Installation Guide](https://cli.github.com/manual/gh_installation)  
  Authenticate with:
  ```bash
  gh auth login
  ```

---

## 2. Initial Setup Script (`setup_ai_coder_project.sh`)

### Step 1: Download the Script
```bash
git clone https://github.com/your-username/AI-Coder-Meta-Agent.git
cd AI-Coder-Meta-Agent/
chmod +x scripts/setup_ai_coder_project.sh
```

### Step 2: Run the Script
```bash
# Basic Usage
./scripts/setup_ai_coder_project.sh

# With Custom Project Name
./scripts/setup_ai_coder_project.sh --project-name my-trading-bot-project

# With Custom Repository and Branch
./scripts/setup_ai_coder_project.sh \
  --project-name my-trading-bot-project \
  --ai-coder-repo https://github.com/your-username/your-ai-coder-fork.git \
  --ai-coder-branch dev

# Help
./scripts/setup_ai_coder_project.sh --help
```

---

## 3. Post-Setup Configuration

After the setup, navigate into your new project:
```bash
cd <YOUR_PROJECT_NAME>_monorepo
```

### Add API Keys to `.env`
Replace placeholders with your actual keys:
```dotenv
GEMINI_API_KEY="your-gemini-key"
OPENAI_API_KEY="your-openai-key"
DEEPSEEK_API_KEY="your-deepseek-key"
```

---

## 4. Running the AI Coder Services

### Step 1: Build Services
```bash
docker compose build
```

### Step 2: Start Services
```bash
docker compose up -d
```

### Step 3: Verify
```bash
docker compose ps
```

---

## 5. Interacting with the AI Coder

### Monitor Progress
```bash
docker compose logs -f orchestration_service
```

### Provide New Goals
Add `.md` or `.json` files to:
```
./<YOUR_PROJECT_NAME>/features/high_level_goals/
```

### Review AI Output
- `services/`: Generated microservices
- `features/bdd/`: Behavior-driven tests
- `features/ai_generated_subtasks/`: Sub-design cards
- `infrastructure/`: Deployment files

---

## 6. Basic Troubleshooting

- **Missing Docker?**
  Make sure Docker Desktop is installed and running.

- **Service crash?**
  Check logs for errors:
  ```bash
  docker compose logs <service_name>
  ```

- **AI not reacting to changes?**
  - Ensure file is in `features/high_level_goals/`
  - File must have `.md` or `.json` extension
  - Check `file_watcher_service` and `orchestration_service` logs

---

