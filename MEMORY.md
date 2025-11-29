# MEMORY.md

This file contains conversation history and context that should be preserved across Claude Code sessions.

## Session: 2025-11-29

### Initial Repository Setup

**User Request**: Analyze codebase and create CLAUDE.md

**Actions Taken**:
- Created initial CLAUDE.md file for newly initialized repository
- Repository identified as AI/ML project using Python

**User Update**: Added `.tool-versions` file with asdf configuration
- Python 3.12.12
- Task 3.45.5

**Actions Taken**:
- Updated CLAUDE.md to reflect correct Python version (3.12.12, not 3.13)
- Documented asdf as version manager
- Documented Task as task runner
- Added setup instructions for asdf

**User Request**: Save conversations in MEMORY.md and update CLAUDE.md to reference it

**Current Status**:
- Repository is newly initialized with no commits yet
- Uses asdf for version management
- Uses Task for task running
- PyCharm/IntelliJ IDEA is the IDE
- No source code added yet
- No Taskfile.yml created yet

**Key Decisions**:
- Using asdf for tool version management
- Using Task as task runner
- Python 3.12.12 as the language version

**Next Steps** (potential):
- Create Taskfile.yml with common tasks
- Set up initial Python project structure
- Add dependencies management (requirements.txt or pyproject.toml)
- Add source code directories and initial modules

### Machine Learning Student Project (2025-11-29)

**User Context**:
- Student working on ML project requiring EDA and AFC (Correspondence Analysis)
- Dataset: Higher education student retention from Kaggle
- Dataset location: `data/dataset.csv`

**Project Requirements**:
- Create Jupyter notebook for analysis
- Perform EDA (Exploratory Data Analysis)
- Perform AFC using `fanalysis` library (NOT prince)

**Key Decisions**:
- Will use `fanalysis` library for AFC analysis
- Main work will be done in Jupyter notebook
- Student prefers to do the work themselves to learn
- Claude's role: Advisory and interpretation support

**Approach**:
- Student will create and work in notebook independently
- Student will share notebook (`@notebooks/filename.ipynb`) when ready for interpretation help
- Can also share visualizations (PNG/JPG) or CSV results for specific questions

### Project Setup Completed (2025-11-29)

**Actions Taken**:
- Created comprehensive project structure with pyproject.toml
- Set up Taskfile.yml with task runner commands
- Added dependencies: pandas, matplotlib, seaborn, scikit-learn, fanalysis
- Configured uv for fast package management
- Created README.md with full documentation
- Added .actrc for act (GitHub Actions local testing) with linux/amd64 architecture
- Configured pre-commit hooks with ruff, isort, and nbstripout
- All pre-commit hooks tested and passing

**Development Tools Configured**:
- Task runner with commands: venv, install, add, add-dev, pre-commit-install, pre-commit-run
- Pre-commit hooks for code quality
- Act for local GitHub Actions testing
- Uv for fast dependency management

**Current Project State**:
- Fully configured Python project ready for ML/AI work
- All documentation complete and up to date
- Pre-commit hooks installed and functional
- Ready for student to create Jupyter notebooks for EDA and AFC analysis
