# GIT COMMIT CHECKLIST - SCARLET SAILS

**Purpose:** Ensure clean, safe commits every time
**Last Updated:** 2025-11-10

---

## ⚡ QUICK COMMIT (When in hurry)

```powershell
# 1. Check status
git status

# 2. Add files
git add .

# 3. Commit with message
git commit -m "Your commit message here"

# 4. Push to branch
git push -u origin claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH
```

**⚠️ Only use quick commit if:**
- Working alone on feature branch
- No sensitive data
- Small changes
- Tested code

---

## ✅ FULL COMMIT CHECKLIST (Recommended)

### STEP 1: Pre-Commit Checks

```
☐ Code runs without errors
☐ No sensitive data (API keys, passwords, .env)
☐ No large files >10MB (except trained models)
☐ No .parquet data files (should be in .gitignore)
☐ Updated documentation if needed
☐ Tested on clean environment (optional but recommended)
```

### STEP 2: Check Git Status

```powershell
# See what will be committed
git status

# See detailed diff
git diff

# See list of files
git ls-files -m  # Modified
git ls-files -o  # Untracked
```

**Look for:**
- ❌ data/raw/*.parquet (should NOT be committed)
- ❌ data/features/*.parquet (should NOT be committed)
- ❌ .env files
- ❌ Temporary files (*.tmp, *.cache)
- ✅ Python code (.py)
- ✅ Documentation (.md)
- ✅ Reports (small CSVs, JSONs)

### STEP 3: Stage Files Selectively

**Option A: Add all (if confident):**
```powershell
git add .
```

**Option B: Add specific files (safer):**
```powershell
# Add specific file
git add models/hybrid_position_manager.py

# Add directory
git add scripts/

# Add by pattern
git add *.md

# Check what's staged
git diff --staged
```

### STEP 4: Write Good Commit Message

**Format:**
```
[type]: [short description]

[optional detailed description]

[optional breaking changes or notes]
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance

**Examples:**
```
Good:
✅ "feat: Add master comprehensive audit script"
✅ "fix: Correct HybridPositionManager API calls"
✅ "docs: Update DAY12_FINAL_SUMMARY with honest assessment"

Bad:
❌ "update"
❌ "changes"
❌ "asdfasdf"
```

### STEP 5: Commit

```powershell
# Commit with message
git commit -m "feat: Your commit message"

# Or open editor for longer message
git commit

# Check commit was created
git log --oneline -1
```

### STEP 6: Push to Remote

**⚠️ IMPORTANT:** Always push to the correct branch!

```powershell
# Push to feature branch (correct!)
git push -u origin claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH

# ❌ NEVER push to main without approval!
# ❌ git push origin main
```

**If push fails:**

```powershell
# Conflict with remote - pull first
git pull origin claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH

# Resolve conflicts if any
# Then push again
git push -u origin claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH
```

**If push rejected (403 error):**
```
Причина: Неправильное название ветки
Решение: Используй полное имя ветки:
         claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH
```

### STEP 7: Verify on GitHub

```powershell
# Open GitHub in browser
# Go to: https://github.com/Antihrist-star/scarlet-sails
# Switch to branch: claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH
# Verify files are there
```

---

## 🚫 NEVER COMMIT THESE

```
❌ .env                                    # API keys, secrets
❌ config.yaml (if contains secrets)       # Configuration with passwords
❌ data/raw/*.parquet                      # Large data files (use .gitignore)
❌ data/features/*.parquet                 # Large feature files
❌ data/processed/*.parquet                # Large processed files
❌ *.pkl (large files)                     # Large pickle files
❌ *.pt (large files)                      # Large PyTorch models
❌ *.log                                   # Log files
❌ __pycache__/                            # Python cache
❌ .pytest_cache/                          # Test cache
❌ .DS_Store                               # Mac system file
❌ Thumbs.db                               # Windows system file
❌ *.swp, *.swo                            # Vim temp files
```

---

## ✅ ALWAYS COMMIT THESE

```
✅ *.py                                    # Python source code
✅ *.md                                    # Documentation
✅ requirements.txt                        # Dependencies
✅ .gitignore                              # Git ignore rules
✅ configs/*.yaml                          # Configuration (non-secret)
✅ Small CSVs (<1MB)                       # Reports, results
✅ Small JSONs (<1MB)                      # Metadata, results
✅ Trained models (<50MB)                  # XGBoost, scalers
✅ README.md                               # Project overview
```

---

## 🔧 COMMON SCENARIOS

### Scenario 1: "I committed a large file by accident"

```powershell
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Remove large file from staging
git reset data/raw/large_file.parquet

# Add to .gitignore
echo "data/raw/*.parquet" >> .gitignore

# Commit again without large file
git add .
git commit -m "Your message"
```

### Scenario 2: "I committed secrets (.env file)"

```powershell
# ⚠️ CRITICAL: Remove from history immediately!

# If not pushed yet:
git reset --soft HEAD~1
git reset .env
git add .
git commit -m "Your message"

# If already pushed:
# 1. Delete .env from repo
git rm --cached .env
echo ".env" >> .gitignore
git add .gitignore
git commit -m "Remove sensitive file"
git push

# 2. Rotate all secrets immediately!
# 3. Consider using git-filter-branch to remove from history
```

### Scenario 3: "My commit message has a typo"

```powershell
# If not pushed yet:
git commit --amend -m "New corrected message"

# If already pushed:
# Don't bother, just make new commit with fix
```

### Scenario 4: "I want to undo changes"

```powershell
# Undo all uncommitted changes (DANGEROUS!)
git reset --hard HEAD

# Undo changes to specific file
git checkout -- filename.py

# Undo last commit but keep changes
git reset --soft HEAD~1

# Undo last commit and discard changes (DANGEROUS!)
git reset --hard HEAD~1
```

### Scenario 5: "Branch is out of sync with remote"

```powershell
# Pull latest from remote
git pull origin claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH

# If conflicts, resolve manually then:
git add .
git commit -m "Merge remote changes"
git push
```

### Scenario 6: "I need to switch branches"

```powershell
# Check current branch
git branch

# See all branches
git branch -a

# Switch to existing branch
git checkout branch-name

# Create new branch and switch
git checkout -b new-branch-name

# Return to previous branch
git checkout -
```

---

## 📋 DETAILED WORKFLOW

### Full Workflow (Step-by-step):

1. **Start of work session:**
   ```powershell
   cd C:\Users\Dmitriy\scarlet-sails
   git status
   git pull origin claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH
   ```

2. **Make changes:**
   - Edit files
   - Test code
   - Update documentation

3. **Check changes:**
   ```powershell
   git status
   git diff
   ```

4. **Stage files:**
   ```powershell
   # Review each file
   git diff filename.py

   # Add if good
   git add filename.py

   # Or add all
   git add .
   ```

5. **Review staged changes:**
   ```powershell
   git diff --staged
   ```

6. **Commit:**
   ```powershell
   git commit -m "feat: Description of changes"
   ```

7. **Push:**
   ```powershell
   git push -u origin claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH
   ```

8. **Verify on GitHub:**
   - Open browser
   - Go to repo
   - Switch to branch
   - Check files

9. **End of session:**
   ```powershell
   git status  # Should be clean
   git log --oneline -3  # Review recent commits
   ```

---

## 🎯 QUICK REFERENCE

### Status Check:
```powershell
git status              # What's changed
git log --oneline -5    # Recent commits
git diff                # See changes
git branch              # Current branch
```

### Common Operations:
```powershell
git add .               # Stage all
git add filename        # Stage specific
git commit -m "msg"     # Commit
git push                # Push to remote
git pull                # Pull from remote
```

### Safety:
```powershell
git diff                # Before staging
git diff --staged       # Before committing
git log -1              # After committing
```

### Undo:
```powershell
git reset HEAD file     # Unstage file
git checkout -- file    # Discard changes
git reset --soft HEAD~1 # Undo commit (keep changes)
```

---

## 🔐 SECURITY REMINDERS

### Before EVERY commit:

```
☐ Run: git status
☐ Check for .env files
☐ Check for API keys in code
☐ Check for large data files
☐ Check .gitignore is working
```

### If you find secrets:

1. **DON'T COMMIT THEM!**
2. Add to .gitignore immediately
3. If already committed, remove from history
4. Rotate all exposed secrets

---

## 📞 HELP

### If stuck:
```powershell
# See what git suggests
git status

# See full help
git help

# See command help
git help commit
git help push
```

### Common errors:

**"error: failed to push"**
- Pull first: `git pull origin branch-name`
- Resolve conflicts if any
- Push again

**"error: src refspec does not match any"**
- Branch name is wrong
- Check: `git branch`
- Use correct name

**"fatal: refusing to merge unrelated histories"**
- Use: `git pull --allow-unrelated-histories`

---

## ✅ FINAL CHECKLIST

Before closing terminal:

```
☐ All changes committed
☐ All commits pushed to remote
☐ Verified on GitHub
☐ No uncommitted changes (git status clean)
☐ Branch is up to date with remote
```

---

**Remember:**
- Commit often (small commits are better)
- Write clear messages
- Never commit secrets
- Always verify on GitHub
- When in doubt, ask!

---

**Branch:** claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH
**Repo:** https://github.com/Antihrist-star/scarlet-sails

---

*Last Updated: Day 12 (2025-11-10)*
