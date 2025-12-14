---
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*), Read, Grep, Glob
description: Create a git commit with automatic message generation
model: claude-haiku-4-5
---
# Claude Command: Commit

This command helps you create well-formatted commits with conventional commit messages and emoji.

## Usage

To create a commit, just type:
```
/commit
```

## What This Command Does

1. Checks which files are staged with `git status`
2. If no files are staged, automatically adds all modified and new files with `git add`
3. Performs a `git diff` to understand what changes are being committed
4. Analyzes the diff to determine if multiple distinct logical changes are present
5. If multiple distinct changes are detected, suggests breaking the commit into multiple smaller commits
6. For each commit (or the single commit if not split), creates a commit message using emoji conventional commit format

## Best Practices for Commits

- **Atomic commits**: Each commit should contain related changes that serve a single purpose
- **Split large changes**: If changes touch multiple concerns, split them into separate commits
- **Conventional commit format**: Use the format `emoji <type>: <description>`
- **Present tense, imperative mood**: Write commit messages as commands (e.g., "add feature" not "added feature")
- **Concise first line**: Keep the first line under 72 characters
- **CRITICAL - No attribution**: NEVER add "Generated with Claude Code", "Co-Authored-By: Claude", or any similar attribution to commits

## Commit Types and Emojis

Use ONE emoji per commit based on the primary type of change:

- ✨ `feat`: New feature or functionality
- 🐛 `fix`: Bug fix (non-critical)
- 🚑️ `fix`: Critical hotfix
- 📝 `docs`: Documentation changes
- 🎨 `style`: Code structure/formatting improvements
- ♻️ `refactor`: Code refactoring (no behavior change)
- 🚚 `refactor`: Move or rename files/resources
- ⚡️ `perf`: Performance improvements
- ✅ `test`: Add or update tests
- 🔧 `chore`: Configuration, tooling, maintenance
- 🔥 `chore`: Remove code or files
- 📦️ `chore`: Update dependencies or packages
- ➕ `chore`: Add a dependency
- ➖ `chore`: Remove a dependency
- 🚀 `ci`: CI/CD changes
- 💚 `fix`: Fix CI build
- 🔒️ `fix`: Security fixes
- ♿️ `feat`: Accessibility improvements
- 🗃️ `chore`: Database migrations or schema changes
- 🌐 `feat`: Internationalization/localization changes

## Guidelines for Splitting Commits

When analyzing the diff, consider splitting commits based on these criteria:

1. **Different concerns**: Changes to unrelated parts of the codebase
2. **Different types of changes**: Mixing features, fixes, refactoring, etc.
3. **File patterns**: Changes to different types of files (e.g., source code vs documentation)
4. **Logical grouping**: Changes that would be easier to understand or review separately
5. **Size**: Very large changes that would be clearer if broken down

## Examples

**Good commit messages for this AI/ML project:**
- ✨ feat: add BERT model for disease classification
- ✨ feat: implement symptom similarity search with Jaccard index
- 🐛 fix: resolve data preprocessing pipeline error
- 🐛 fix: correct GPU memory allocation in training script
- 📝 docs: update CLAUDE.md with Lightning Studio tasks
- ♻️ refactor: extract data preprocessing to separate module
- ♻️ refactor: simplify model evaluation metrics calculation
- 🎨 style: improve notebook cell organization for EDA
- 🔥 chore: remove deprecated baseline model code
- 📦️ chore: update transformers to 4.36.0
- ➕ chore: add lightning-ai dependency
- ➖ chore: remove unused scikit-learn dependency
- 🚀 ci: add model training workflow
- ⚡️ perf: optimize dataset loading with caching
- ✅ test: add unit tests for feature engineering

**Example of splitting commits:**

If you modify both training script AND add new task commands, split into:
1. ✨ feat: add T4 GPU support for Lightning Studio
2. 🔧 chore: add Lightning Studio management tasks to Taskfile

If you work on multiple unrelated improvements, split into:
1. 🐛 fix: resolve tokenizer padding issue in BERT model
2. ⚡️ perf: optimize symptom vectorization performance
3. 📝 docs: document model evaluation methodology

## Important Notes

- If specific files are already staged, the command will only commit those files
- If no files are staged, it will automatically stage all modified and new files
- The commit message will be constructed based on the changes detected
- Before committing, the command will review the diff to identify if multiple commits would be more appropriate
- If suggesting multiple commits, it will help you stage and commit the changes separately
