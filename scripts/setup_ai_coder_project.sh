#!/bin/bash

# --- Configuration Variables (can be overridden by arguments) ---
AI_CODER_REPO_URL="https://github.com/your-org/AI-Coder-Meta-Agent.git"
AI_CODER_BRANCH="main"
DEFAULT_PROJECT_NAME="my-ai-project"

# --- Colors ---
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- Usage ---
display_usage() {
    echo -e "${GREEN}Usage: $0 [--project-name <name>] [--ai-coder-repo <url>] [--ai-coder-branch <branch>]${NC}"
    echo "Example: $0 --project-name my-trading-bot --ai-coder-repo https://github.com/myuser/ai-coder.git"
    exit 1
}

# --- Argument Parser ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --project-name) PROJECT_NAME="$2"; shift ;;
        --ai-coder-repo) AI_CODER_REPO_URL="$2"; shift ;;
        --ai-coder-branch) AI_CODER_BRANCH="$2"; shift ;;
        -h|--help) display_usage ;;
        *) echo -e "${RED}Unknown parameter: $1${NC}"; display_usage ;;
    esac
    shift
done

# --- Defaults ---
if [ -z "$PROJECT_NAME" ]; then
    PROJECT_NAME="$DEFAULT_PROJECT_NAME"
    echo -e "${YELLOW}No project name provided. Using default: ${PROJECT_NAME}${NC}"
fi

AI_CODER_DIR="AI-Coder-Meta-Agent"

# --- 1. Prerequisite Check ---
echo -e "${GREEN}1. Checking prerequisites...${NC}"
command -v git >/dev/null || { echo -e "${RED}Git not installed${NC}"; exit 1; }
command -v docker >/dev/null || { echo -e "${RED}Docker not installed${NC}"; exit 1; }
if command -v docker-compose >/dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
elif docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    echo -e "${RED}Docker Compose not found${NC}"; exit 1
fi
echo -e "${GREEN}   Prerequisites met.${NC}"

# --- 2. Create Monorepo Folder ---
PARENT_DIR="${PROJECT_NAME}_monorepo"
echo -e "${GREEN}2. Creating monorepo: $PARENT_DIR${NC}"
mkdir -p "$PARENT_DIR" && cd "$PARENT_DIR" || exit 1

# --- 3. Clone AI Coder Repo ---
echo -e "${GREEN}3. Cloning AI Coder...${NC}"
git clone --branch "$AI_CODER_BRANCH" "$AI_CODER_REPO_URL" "$AI_CODER_DIR" || exit 1

# --- 4. Create Project Structure ---
echo -e "${GREEN}4. Setting up project structure...${NC}"
mkdir -p "$PROJECT_NAME"/{services,common,infrastructure,docs}
mkdir -p "$PROJECT_NAME"/features/{high_level_goals,ai_generated_subtasks,detailed_specifications,components_and_modules,characterization,bug_fixes,refactoring_tasks,bdd}

# README
cat <<EOF > "$PROJECT_NAME/README.md"
# $PROJECT_NAME

This is the main project managed by the AI Coder meta-agent.

### Features
* High-Level Goals: \`features/high_level_goals/\`
* AI-Generated Subtasks: \`features/ai_generated_subtasks/\`
* Application Services: \`services/\`
* Test Code: \`services/*/tests/\`
* BDD Features: \`features/bdd/\`
* Deployment Infrastructure: \`infrastructure/\`
EOF

# Requirements
cat <<EOF > "$PROJECT_NAME/requirements.txt"
# Top-level requirements
Flask
pytest
EOF

# Feature README
cat <<EOF > "$PROJECT_NAME/features/ai_generated_subtasks/README.md"
# AI Generated Subtasks
Do not manually modify files in this directory unless you understand the AI's workflow.
EOF

cat <<EOF > "$PROJECT_NAME/infrastructure/README.md"
# Infrastructure
Deployment configurations for the AI project (e.g., Docker, Kubernetes).
EOF

# System Overview
cat <<EOF > "$PROJECT_NAME/features/high_level_goals/system_overview.md"
# System Overview: My Trading Bot

## Purpose
Develop an AI trading bot using microservices.

## Core Features
1. Market Data Fetcher
2. Strategy Executor
3. Order Placement

## Architecture
- REST APIs
- Docker Compose
EOF

# --- 5. Docker Compose ---
echo -e "${GREEN}5. Creating docker-compose.yaml...${NC}"
cat <<EOF > docker-compose.yaml
services:
  orchestration_service:
    image: ai-coder-orchestration
    build: ./$AI_CODER_DIR
    ports:
      - "5006:5006"
    volumes:
      - ./$PROJECT_NAME:/app
    networks:
      - agentic-net
networks:
  agentic-net:
    driver: bridge
EOF

# --- 6. .env File ---
echo -e "${GREEN}6. Creating .env file...${NC}"
cat <<EOF > .env
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY"
EOF

# --- 7. Final Tips ---
echo -e "${GREEN}--- Setup Complete! ---${NC}"
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. cd $PARENT_DIR"
echo "2. Edit the .env file with actual API keys"
echo "3. Run: $DOCKER_COMPOSE_CMD build && $DOCKER_COMPOSE_CMD up -d"
echo "4. Monitor with: $DOCKER_COMPOSE_CMD logs -f orchestration_service"
