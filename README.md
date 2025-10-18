# Machine Learning: Classification and Regression Analysis

A comprehensive machine learning project implementing gradient descent for binary classification and neural network regression for real-world datasets.

## Project Overview

This project demonstrates proficiency in fundamental machine learning concepts through two distinct analyses:

1. **Loan Approval Classification**: From-scratch implementation of linear classification using gradient descent with hinge loss and L2 regularization
2. **Agricultural Yield Prediction**: Multi-layer perceptron (MLP) regression for soybean production forecasting

## Technical Highlights

### Linear Classification Implementation

- Custom gradient descent algorithm for binary classification
- Hinge loss with L2 regularization: $O = C \sum^N_{i=1}\max(0, 1 - y_i (\mathbf{w}^T\mathbf{x}_i + w_0)) + \frac{1}{2}\mathbf{w}^T\mathbf{w}$
- Vectorized NumPy operations for computational efficiency
- Hyperparameter analysis examining learning rate effects on convergence

### MLP Regression Analysis

- Grid search optimization across multiple architectures and hyperparameters
- Cross-validation for robust model selection
- Feature engineering including one-hot encoding and standardization
- Performance evaluation using MSE and R² metrics

## Datasets

### Loan Approval Dataset

- **Domain**: Financial services
- **Task**: Binary classification (approve/reject)
- **Features**: Personal demographics, financial history, loan characteristics
- **Size**: Multi-thousand instances with mixed categorical/numerical features

### Soybean Agricultural Dataset

- **Domain**: Agricultural technology
- **Task**: Regression (yield prediction)
- **Features**: Genotype, treatments, environmental conditions, plant characteristics
- **Size**: 52,678 instances with 13 features

## Key Results

### Classification Performance

- **Training Accuracy**: 85%+ achieved through optimized gradient descent
- **Learning Rate Analysis**: Comprehensive study showing optimal convergence rates
- **Regularization Impact**: Demonstrated effective overfitting prevention

### Regression Performance

- **Best Architecture**: (100, 100) hidden layers with ReLU activation
- **R² Score**: Strong predictive performance on agricultural yield
- **Model Selection**: Systematic comparison of architectures and hyperparameters

## Technical Implementation

### Core Technologies

- **NumPy**: Vectorized mathematical operations
- **Pandas**: Data manipulation and preprocessing
- **Scikit-learn**: MLP implementation and evaluation metrics
- **Matplotlib**: Professional data visualization

### Key Features

- No black-box implementations: Linear classifier built from mathematical foundations
- Efficient algorithms: Avoided loops in favor of vectorized operations
- Reproducible results: Fixed random seeds and documented parameters
- Professional visualizations: Clear plots with proper labels and formatting

## Project Structure

```
machine-learning-classification-regression/
├── README.md                      # Project documentation
├── machine_learning_analysis.ipynb # Main analysis notebook
├── requirements.txt               # Python dependencies
├── data/                         # Datasets
│   ├── loan_data.csv
│   └── soybean_data.csv
├── src/                          # Source code
│   └── model_eval_utils.py
└── models/                       # Trained models
    └── best_mlp_model.sav
```

## Setup and Usage

### Installation

```bash
git clone https://github.com/yourusername/machine-learning-classification-regression.git
cd machine-learning-classification-regression
pip install -r requirements.txt
```

### Running the Analysis

```bash
jupyter notebook machine_learning_analysis.ipynb
```

### Dependencies

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- OpenPyXL

## Performance Insights

### Classification Learnings

- **Learning rate sensitivity**: Optimal rates in 0.0001-0.001 range
- **Convergence patterns**: Smooth objective function decrease indicates proper implementation
- **Generalization**: Minimal overfitting with appropriate regularization

### Regression Findings

- **Architecture importance**: Deeper networks (100, 100) significantly outperformed shallow ones
- **Activation functions**: ReLU consistently outperformed logistic activation
- **Feature engineering**: Proper encoding of categorical variables crucial for performance

## Educational Value

This project demonstrates:

- Mathematical understanding: Implementation from first principles
- Software engineering: Clean, efficient, and documented code
- Experimental design: Systematic hyperparameter analysis
- Real-world application: Practical problems in finance and agriculture
- Performance optimization: Vectorized operations and efficient algorithms

## Future Enhancements

- Cross-validation for linear classifier
- Additional regularization techniques (L1, Elastic Net)
- Feature importance analysis
- Model interpretability visualizations
- Comparison with ensemble methods

## Academic Context

This project was completed as part of COMP24112 (Machine Learning) coursework, demonstrating mastery of:

- Linear classification theory and implementation
- Neural network regression techniques
- Experimental methodology in machine learning
- Professional software development practices

---

**Note**: This project emphasizes understanding over black-box usage, implementing core algorithms from mathematical foundations while following industry best practices for code quality and documentation.
