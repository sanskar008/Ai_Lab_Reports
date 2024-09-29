install.packages("bnlearn")
#  Q.1

# Read the data from the file
library(bnlearn)

# Step 1: Read Data
data.grades <- read.table("2020_bn_nb_data.txt", header = TRUE)

# Convert variables to factors
data.grades <- lapply(data.grades, as.factor)
data.grades <- data.frame(data.grades)

# Step 2: Learn Dependencies (Bayesian Network Structure)
data.grades.net <- hc(data.grades[, -9], score = 'k2')
print (data.grades.net)

#  Q.2

library(bnlearn)

# Step 1: Read Data
data.grades <- read.table("2020_bn_nb_data.txt", header = TRUE)

# Convert variables to factors
data.grades <- lapply(data.grades, as.factor)

# Exclude the QP node from the data
data.grades excluded_qp <- data.grades[, -9]

#  Step 2: Learn Dependencies (Bayesian network Structure)
data.grades.net <- hc(data.grades excluded_qp, score = 'k2')

# Plot Bayesian Network Structure
plot(data.grades.net)

# Learn the conditional probability Tables (CPTS) for each node in the Bayesian network
cpt <- bn.fit(data.grades.net, data = data.grades excluded_qp)

# print the CPTS for each course node
print(cpt)

#  Q.3

install.packages("e1071")
install.packages("pomegranate")

library(e1071)

# Step 1: Read Data
data.grades <- read.table("2020_bn_nb_data.txt", header = TRUE)

# Convert variables to factors
data.grades <- lapply(data.grades, as.factor)

# Define the variables for prediction
EC100 <- "DD"
IT101 <- "CC"
MA101 <- "CD"

# Train the Naive gayes (lassifier
nb_classifier <- nb_classifier(PH100 ~ EC100 + IT101 + MA101, data = data)

# Create new data frame for prediction
new_data <- data.frame(EC100 = factor(EC100, levels = levels(data$EC100)),
                       EC100 = factor(IT101, levels = levels(data$IT101)),
                       MA101 = factor(MA101, levels = levels(data$MA101)))

# predict the grade in
prediction <- predict( nb_classifier, newdata = new_data)

# print the predicted grade
print(prediction)


# Q.4
# Load the required library
library(e1071)

# Step 1: Read Data
data.grades <- read.table("2020_bn_nb_data.txt", header = TRUE)

# Function to split data into training and testing sets
split data function(data, train _ size) {
n nrow(data)
train_indices size = floor(train_size * n))
train_data <- data[train_indices,
test_data data[ -train _ indices,
= train_data, test_data = test_data))

# Function to train and test the naive Bayes classifier
train_and_test_nb test_data <- function(train_data, test_data) {

# Train naive Bayes classifier
nb_classifier <- naiveBayes(QP ~ ., data = train data)

# Predict on test dat
predictions <- predict(nb_classifier, test_data[, -ncol(test_data)])

# calculate accuracy
accuracy <- sum(predictions == test_data$QP) / nrow(test_data)

return (accuracy)
}

# Number of iterations
num iterations <- 20

# Store accuracies
accuracies <- numeric(num_iterations)

# Run experiments
for (i in l:num_iterations){
    # Split data into training and testing sets
    split <- split_data(data, 0.7)

    # Train and test naive Bayes classifier
    accuracies[i] train_and_test_nb(split$train_data, split$test_data)
}

# Report results
cat("Mean accuracy:" , mean(accuracies),"\n")
cat("standard deviation of accuracy:", sd(accuracies) ,"\n")



#  Q.5

# Load the required library
library(bnlearn)

# Step 1: Read Data
data.grades <- read.table("2020_bn_nb_data.txt", header = TRUE)

# Function to split data into training and testing sets
split_data function(data, train_size) {
    n <- nrow(data)
    train_indices <- sample(1:n, size = floor(train size * n))
    train_data <- data[train_indices, ]
    test_data <- data[-train_indices, ]
    return(list(train_data = train_data, test_data = test_data))
}

# Function to train and test the Bayesian network classifier
train_and_test_bn <- function(train_data, test_data) {
    # Learn the structure of the Bayesian network using Hill-Climbing with BIC score
     bn_structure <- hc(train_data, score = "bic")

    # Handle level mismatch in PH160:
    # Identify valid levels in the data and training set
    valid_levels <- intersect(levels(train_data$PH160), levels(test_data$PH160))

    # Remove unused levels
    train_data$PH160 <- factor(train_data$PH160, levels = valid_levels)
    trest_data$PH160 <- factor(train_data$PH160, levels = valid_levels)

    # Learn the Conditional Probability Tables (CPTs) for each node
    bn_params <- bn.fit(bn_structure, train_data)

   # Predict on test data
   predictions <- predict(bn_params, node = "QP", method = "bayes -lw", data = test_data)

   # Calculate accuracy
   accuracy <- sum(predictions == test_data$QP) / nrow(test_data)

   return (accuracy)
}


# Number of iterations
num iterations <- 20

# Store accuracies
accuracies <- numeric(num_iterations)

# Run experiments
for (i in l:num_iterations){
    # Split data into training and testing sets
    split <- split_data(data, 0.7)

    # Train and test naive Bayes classifier
    accuracies[i] train_and_test_nb(split$train_data, split$test_data)
}

# Report results
cat("Mean accuracy:" , mean(accuracies),"\n")
cat("standard deviation of accuracy:", sd(accuracies) ,"\n")