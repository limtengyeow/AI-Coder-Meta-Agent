    # AI-Coder-Meta-Agent

    [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
    [![GitHub Workflow Status](https://img.shields.io/github/workflow/status/limtengyeow/AI-Coder-Meta-Agent/main?label=build&logo=github)](https://github.com/limtengyeow/AI-Coder-Meta-Agent/actions)
    <!-- Add more badges here if you set up CI/CD, etc. -->

    An autonomous AI agent for software development, designed to streamline the entire process from high-level goals to deployed applications. Leveraging Large Language Models (LLMs) and a rigorous Test-Driven Development (TDD) workflow, this meta-agent can autonomously:

    * Decompose complex features into granular tasks.
    * Generate application code for microservices.
    * Create comprehensive unit and integration tests.
    * Perform automated code reviews and suggest fixes.
    * Generate Dockerfiles and deployment configurations (e.g., Docker Compose for user apps, Kubernetes manifests).
    * Drive development incrementally through a Red-Green-Refactor cycle.

    This project aims to demonstrate and facilitate a truly autonomous software development experience, keeping human developers focused on architecture and high-level strategy.

    ---

    ## 🚀 Getting Started: Set Up a New AI-Managed Project

    To begin using the AI Coder Meta-Agent and create your first AI-managed software project, please follow our comprehensive step-by-step setup guide:

    [**📖 View Setup Guide: docs/setup_guide.md**](docs/setup_guide.md)

    The setup guide will walk you through:
    * System prerequisites (Git, Docker).
    * Running the automated setup script.
    * Configuring your LLM API keys.
    * Building and starting the AI Coder services.
    * How to interact with the AI Coder by providing high-level design goals.

    ---

    ## 🌐 Project Structure

    This repository itself contains the core services of the AI Coder Meta-Agent. When you set up a new project using the guide, it will create a structured workspace for your AI-managed application:

    ```
    my-trading-bot_monorepo/
    ├── AI-Coder-Meta-Agent/   # This repository's code (the AI Coder itself)
    ├── my-new-application/    # Your AI-managed project code
    │   ├── services/          # Microservices (AI-generated)
    │   ├── features/          # Design goals, BDD, AI-generated subtasks
    │   └── infrastructure/    # Deployment configs (AI-generated)
    └── docker-compose.yaml    # Top-level Docker orchestration
    └── .env                   # LLM API keys
    ```

    ---

    ## ✨ Key Features & Capabilities

    * **Test-Driven Development (TDD):** Automated Red-Green-Refactor loop.
    * **Modular Architecture:** Designed for microservices development.
    * **Dynamic Prompting:** LLMs intelligently generate prompts for sub-tasks.
    * **Autonomous Debugging:** Learns to fix common errors based on test and review feedback.
    * **Code Quality Enforcement:** Automated code reviews ensure best practices.
    * **BDD Integration:** Drives development from behavioral specifications.
    * **Deployment Artifact Generation:** From code to containerization.

    ---

    ## 🤝 Contributing

    We welcome contributions to improve the AI Coder Meta-Agent! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) (if you create one) for guidelines.

    ---

    ## 📄 License

    This project is licensed under the [MIT License](LICENSE).
    ```
