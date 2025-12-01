# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Read MEMORY.md First

**Always read `MEMORY.md` when starting a new session** to understand the conversation history, context, and decisions made in previous sessions.

**Periodically update `MEMORY.md`** during your session to preserve important context, decisions, and conversation history for future sessions.

## Project Overview

This is an AI/ML project using Python 3.13. The repository is currently in its initial setup phase.

## Development Environment

- **Python Version**: 3.12.12 (managed via asdf, see `.tool-versions`)
- **Task Runner**: Task 3.45.5 (managed via asdf)
- **Version Manager**: asdf
- **IDE**: PyCharm/IntelliJ IDEA with Python plugin

## Project Structure

The repository is newly initialized. As the project develops, standard Python project structure is expected:
- Source code will likely be organized in a main package directory
- Tests will typically be in a `tests/` directory
- Dependencies will be managed via `requirements.txt` or `pyproject.toml`
- ML/AI specific directories for models, datasets, notebooks, etc. may be added

## Development Commands

### Setup
```bash
# Install correct tool versions
asdf install
```

### Task Runner
This project uses [Task](https://taskfile.dev/) as a task runner. Available tasks will be defined in `Taskfile.yml`.

```bash
# List available tasks
task --list
```

(Additional commands to be populated as the project structure is established)

## Architecture Notes

(To be populated as the architecture evolves)

This section will document:
- Key modules and their responsibilities
- Data flow and processing pipelines
- Model architecture and training procedures
- Integration points between components
