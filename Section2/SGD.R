set.seed(1001)
N <- 10000000
int <- rep(1, N)
x1 <- rnorm(N, mean = 10, sd = 2)
Y <- rnorm(N, 2*x1 + int, 1)
model <- lm(Y ~ x1)
beta <- coef(model)
beta

# Starting points
set.seed(234)
b <- runif(1, 0, 1)
c <- runif(1, 0, 1)
n <- length(x1)

# Parameters
learning_rate <- 0.01
batch <- 10000
epochs <- 10 # Reduced for practicality
epsilon <- 1e-6 # Small threshold for convergence check

# Gradient
yhat <- c + b * x1
MSE <- sum((Y - yhat) ^ 2) / n
converged <- FALSE
iterations <- 0
MSE_change <- numeric(epochs) # Pre-allocate for efficiency

# While loop
while (!converged && iterations < epochs) {
  # Shuffle data points
  indices <- sample(n, n)
  
  for (i in seq(1, n, by = batch)) {
    idx <- indices[i:min(i + batch - 1, n)]
    x_batch <- x1[idx]
    y_batch <- Y[idx]
    
    yhat_batch <- c + b * x_batch
    
    # Gradient calculation
    b_gradient <- -(1 / length(idx)) * sum((y_batch - yhat_batch) * x_batch)
    c_gradient <- -(1 / length(idx)) * sum(y_batch - yhat_batch)
    
    # Update parameters
    b <- b - learning_rate * b_gradient
    c <- c - learning_rate * c_gradient
  }
  
  # Recalculate yhat and MSE
  yhat <- c + b * x1
  MSE_new <- sum((Y - yhat) ^ 2) / n
  MSE_change[iterations + 1] <- abs(MSE_new - MSE)
  MSE <- MSE_new
  
  # Check for convergence
  if (MSE_change[iterations + 1] < epsilon) {
    converged <- TRUE
  }
  
  iterations <- iterations + 1
}

# Trim the unused portion of the pre-allocated vector
MSE_change <- MSE_change[1:iterations]

# Output
list(iterations = iterations, c = c, b = b, MSE_change = tail(MSE_change, 1), beta)

