# Regression Implementations

This repository contains implementations of regression algorithms and related statistical concepts written from scratch.

The purpose of the project is to study the mathematical foundations behind regression models instead of only using high-level machine learning libraries.

Even though regression is commonly associated with machine learning today, regression methods are much older than the field itself. The mathematical foundation of linear regression is strongly connected to the method of least squares, developed in the early 1800s by :contentReference[oaicite:0]{index=0} and independently by Legendre for astronomical and measurement problems.

This project focuses on implementing these ideas directly in code to better understand:

- How parameters are estimated
- Why least squares works
- Optimization methods
- Error minimization
- Statistical interpretation of regression coefficients
- Prediction and evaluation metrics

---

## Implemented Topics

- Simple Linear Regression
- Multiple Linear Regression
- Mean Squared Error (MSE)
- Gradient Descent
- Normal Equation
- Feature Scaling
- Prediction Metrics
- Residual Analysis

---

## Project Structure

```text
.
├── simple_linear_regression.py
├── requirements.txt
└── README.md
```

---

## Running the Project

Clone the repository:

```bash
git clone <repository_url>
cd <repository_name>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python {regression that you want}.py
```

---

## Mathematical Background

In simple linear regression, the objective is to estimate a line that minimizes the squared residuals between the observed values and the predicted values.

The regression equation is:


::contentReference[oaicite:1]{index=1}


Where:

- \( \beta_0 \) is the intercept
- \( \beta_1 \) is the slope coefficient
- \( \varepsilon \) represents the residual error

The least squares estimator minimizes:

:contentReference[oaicite:2]{index=2}

---

## References

- [Least Squares — Wikipedia](https://en.wikipedia.org/wiki/Least_squares?utm_source=chatgpt.com)
- [Linear Regression Notes (Stanford)](https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf?utm_source=chatgpt.com)
- [An Introduction to Statistical Learning](https://www.statlearning.com/?utm_source=chatgpt.com)
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/?utm_source=chatgpt.com)
