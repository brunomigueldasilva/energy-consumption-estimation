# Contributing to Electricity Bill Prediction

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior
- Be respectful and considerate
- Welcome newcomers and help them learn
- Focus on what's best for the project
- Show empathy towards other contributors

### Unacceptable Behavior
- Harassment or discriminatory language
- Personal attacks or trolling
- Publishing others' private information
- Other unprofessional conduct

---

## Getting Started

### Prerequisites
- Python 3.13 or higher
- Git
- Basic understanding of machine learning concepts
- Familiarity with scikit-learn and regression techniques

### Fork and Clone
1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/brunomigueldasilva/energy-consumption-estimation.git
cd energy-consumption-estimation
```

---

## Development Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Development Dependencies
```bash
# Optional development tools
pip install pytest pytest-cov pylint black flake8
```

### 4. Verify Setup
```bash
python 08_orchestrator.py --all
```

---

## How to Contribute

### Types of Contributions

#### 🐛 Bug Reports
- Use the bug report template
- Include Python version, OS, and error messages
- Provide steps to reproduce
- Suggest a fix if possible

#### ✨ Feature Requests
- Explain the problem you're trying to solve
- Describe your proposed solution
- Consider alternative approaches
- Explain why this benefits the project

#### 📝 Documentation
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve docstrings
- Translate documentation

#### 💻 Code Contributions
- Bug fixes
- New features (models, visualizations, etc.)
- Performance improvements
- Code refactoring

---

## Coding Standards

### Python Style Guide
Follow **PEP 8** conventions:

```python
# Good
def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: MAPE percentage
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Bad
def calc_err(y,yp):
    return np.mean(np.abs((y-yp)/y))*100
```

### Code Quality Checklist
- [ ] Follows PEP 8 style guide
- [ ] Includes docstrings for all functions
- [ ] Has meaningful variable names
- [ ] Includes comments for complex logic
- [ ] No hardcoded values (use constants)
- [ ] Handles errors gracefully
- [ ] Is DRY (Don't Repeat Yourself)

### Documentation Strings
Use Google-style docstrings:

```python
def train_regression_model(X_train, y_train, model_type='linear'):
    """
    Train a regression model.
    
    This function trains a specified regression model on the provided
    training data and returns the fitted model object.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        model_type (str, optional): Model type to train. 
            Options: 'linear', 'ridge', 'lasso', 'rf'. Defaults to 'linear'.
    
    Returns:
        object: Fitted scikit-learn model
        
    Raises:
        ValueError: If model_type is not recognized
        
    Example:
        >>> X_train, y_train = load_data()
        >>> model = train_regression_model(X_train, y_train, 'ridge')
        >>> print(model.score(X_test, y_test))
    """
    # Implementation
```

---

## Commit Messages

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Examples
```bash
# Good
feat(models): add XGBoost regressor
fix(preprocessing): correct date parsing for meter readings
docs(readme): update installation instructions

# Bad
fixed stuff
update
changes
```

### Detailed Commit Example
```bash
feat(models): add Random Forest regressor

- Implemented Random Forest with 100 estimators
- Added feature importance extraction
- Updated comparative metrics to include RF
- Added RF to orchestrator pipeline

Closes #23
```

---

## Pull Request Process

### Before Submitting

1. **Create a new branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**

3. **Test thoroughly**:
```bash
# Run the full pipeline
python 08_orchestrator.py --all

# Run specific tests (if available)
pytest tests/
```

4. **Update documentation** if needed

5. **Commit with clear messages**:
```bash
git add .
git commit -m "feat(models): add Gradient Boosting regressor"
```

6. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

### Submitting the PR

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template:
   - **Title**: Clear, concise description
   - **Description**: What changes you made and why
   - **Issue Reference**: Link related issues
   - **Testing**: How you tested your changes
   - **Screenshots**: If applicable (for visualizations)

### PR Review Process

1. **Automated checks** will run (if configured)
2. **Maintainer review** (typically within 3-5 days)
3. **Address feedback** if requested
4. **Approval and merge** by maintainer

### PR Checklist
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] PR description is complete

---

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_preprocessing.py
```

### Writing Tests
```python
import pytest
import pandas as pd
from preprocessing import engineer_cutoff_features

def test_cutoff_feature_engineering():
    """Test that cutoff features are created correctly."""
    df = pd.DataFrame({
        'ano': [2024, 2024],
        'mes': [1, 2],
        'consumo_vazio': [100, 120],
        'consumo_ponta': [50, 60]
    })
    
    result = engineer_cutoff_features(df, cutoff_day=15)
    
    assert 'consumo_ate_dia_15' in result.columns
    assert result['consumo_ate_dia_15'].notna().all()
```

---

## Documentation

### Types of Documentation

#### Code Comments
```python
# Use comments to explain WHY, not WHAT
# Good
# Stratified split maintains temporal order for time series
X_train, X_test = train_test_split(X, shuffle=False)

# Bad
# Split the data
X_train, X_test = train_test_split(X)
```

#### Docstrings
- Every public function must have a docstring
- Include Args, Returns, Raises, and Examples
- Keep them up-to-date with code changes

#### README Updates
- Update README.md when adding major features
- Keep installation instructions current
- Update examples if API changes

#### CHANGELOG
Document all notable changes:
```markdown
## [1.1.0] - 2025-11-15
### Added
- XGBoost regressor model
- Feature importance visualization
- Hyperparameter tuning with GridSearchCV

### Fixed
- Date parsing issue in meter readings
- Memory leak in data loading

### Changed
- Updated scikit-learn to 1.5.0
- Improved error messages
```

---

## Development Workflow

### Typical Workflow
1. **Pick an issue** or create one
2. **Discuss** your approach (for large changes)
3. **Create a branch**
4. **Write code** with tests
5. **Update documentation**
6. **Submit PR**
7. **Address feedback**
8. **Celebrate** when merged! 🎉

### Branch Naming
```bash
feature/add-xgboost-model        # New features
bugfix/fix-date-parsing          # Bug fixes
docs/update-installation         # Documentation
refactor/optimize-preprocessing  # Code improvements
```

---

## Getting Help

### Questions?
- Check existing issues and discussions
- Ask in GitHub Discussions
- Email maintainers (for private matters)

### Stuck?
- Don't be afraid to ask for help
- Provide context and what you've tried
- Be patient and respectful

---

## Recognition

Contributors will be:
- Listed in `AUTHORS.md`
- Mentioned in release notes
- Given credit in the README

Thank you for contributing! 🙏

---

## Additional Resources

- [PEP 8 Style Guide](https://pep8.org/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Scikit-learn Contributing Guide](https://scikit-learn.org/stable/developers/contributing.html)
- [Writing Good Commit Messages](https://chris.beams.io/posts/git-commit/)

---

## Specific Contribution Areas

### 1. Model Improvements
- Add new regression algorithms
- Implement ensemble methods
- Add hyperparameter tuning
- Improve feature engineering

### 2. Visualization Enhancements
- Create interactive plots
- Add dashboard capabilities
- Improve existing visualizations
- Add new diagnostic plots

### 3. Performance Optimization
- Optimize data loading
- Improve memory usage
- Speed up training
- Parallelize computations

### 4. Feature Additions
- Real-time API
- Web interface
- Mobile app integration
- Database connectivity

### 5. Testing & Quality
- Add unit tests
- Integration tests
- Performance benchmarks
- Code coverage improvements

---

**Questions?** Feel free to reach out to the maintainers or open a discussion on GitHub.

**Contact**: bruno_m_c_silva@proton.me
