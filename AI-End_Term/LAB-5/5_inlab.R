
library(bnlearn)
library(dplyr)
library(ggplot2)
library(bnclassify)
library(caret)


set.seed(123)


load_and_preprocess_data <- function(file_path) {
  
  course.grades <- read.table(file_path, header = TRUE)
  
  
  course.grades <- as.data.frame(lapply(course.grades, as.factor))
  
  return(course.grades)
}


learn_bn_structure <- function(data, exclude_col = NULL) {
  
  if (!is.null(exclude_col)) {
    data <- data[, -exclude_col]
  }
  
  
  bn_structure <- hc(data, score = 'k2')
  
  return(bn_structure)
}


fit_bayesian_network <- function(bn_structure, data, exclude_col = NULL) {
  
  if (!is.null(exclude_col)) {
    data <- data[, -exclude_col]
  }
  
  
  bn_fit <- bn.fit(bn_structure, data)
  
  return(bn_fit)
}


visualize_cpds <- function(bn_fit, nodes) {
  
  par(mfrow = c(ceiling(length(nodes)/3), 3))
  
  for (node in nodes) {
    bn.fit.barchart(bn_fit[[node]], main = node)
  }
  
  
  par(mfrow = c(1,1))
}


analyze_cpd <- function(bn_fit, target_node, evidence) {
  
  cpd_data <- data.frame(cpdist(bn_fit, nodes = target_node, evidence = evidence))
  
  
  summary_df <- cpd_data %>%
    group_by(!!sym(target_node)) %>%
    summarise(counts = n(), 
              percentage = n() / nrow(cpd_data) * 100)
  
  
  p <- ggplot(summary_df, aes(x = !!sym(target_node), y = counts)) +
    geom_bar(fill = "
    geom_text(aes(label = sprintf("%d (%.1f%%)", counts, percentage)), 
              vjust = -0.3) +
    labs(title = paste("Conditional Distribution of", target_node),
         x = target_node,
         y = "Count") +
    theme_minimal()
  
  print(p)
  return(summary_df)
}


compare_classification_models <- function(train_data, test_data, target_var) {
  
  nb_model <- nb(class = target_var, dataset = train_data)
  nb_model <- lp(nb_model, train_data, smooth = 0.1)
  
  
  tan_model <- tan_cl(target_var, train_data)
  tan_model <- lp(tan_model, train_data, smooth = 0.1)
  
  
  nb_pred <- predict(nb_model, test_data)
  tan_pred <- predict(tan_model, test_data)
  
  
  nb_cm <- confusionMatrix(nb_pred, test_data[[target_var]])
  tan_cm <- confusionMatrix(tan_pred, test_data[[target_var]])
  
  
  results <- list(
    NaiveBayes = list(
      Accuracy = nb_cm$overall['Accuracy'],
      ConfusionMatrix = nb_cm$table
    ),
    TreeAugmentedNB = list(
      Accuracy = tan_cm$overall['Accuracy'],
      ConfusionMatrix = tan_cm$table
    )
  )
  
  return(results)
}


main <- function() {
  
  course.grades <- load_and_preprocess_data("2020_bn_nb_data.txt")
  
  
  course.grades.net <- learn_bn_structure(course.grades, exclude_col = 9)
  
  
  plot(course.grades.net)
  
  
  course.grades.fit <- fit_bayesian_network(course.grades.net, course.grades, exclude_col = 9)
  
  
  nodes_to_visualize <- c("EC100", "EC160", "IT101", "IT161", "MA101", "PH100", "PH160")
  visualize_cpds(course.grades.fit, nodes_to_visualize)
  
  
  analyze_cpd(course.grades.fit, "PH100", 
              expression(EC100 == "DD" & IT101 == "CC" & MA101 == "CD"))
  
  
  train_indices <- createDataPartition(course.grades$QP, p = 0.75, list = FALSE)
  course.grades.train <- course.grades[train_indices, ]
  course.grades.test <- course.grades[-train_indices, ]
  
  
  model_comparison <- compare_classification_models(
    course.grades.train, 
    course.grades.test, 
    target_var = "QP"
  )
  
  
  print("Model Comparison Results:")
  print(model_comparison)
}


main()