library(ggplot2)


baseline_predictions <- read.csv("./predictions/linres/2023_10_1_15_37_13.csv")
dnn_predictions <- read.csv("./predictions/dnn/gene_predict_IC50_2023_10_17_19_34_05.csv")

baseline_cor <- cor(baseline_predictions$predictions, baseline_predictions$actual)
dnn_cor <- cor(dnn_predictions$predictions, dnn_predictions$actual)

prediction_types = c(rep("Baseline", nrow(baseline_predictions)) , rep("DNN", nrow(dnn_predictions)))

prediction_df <- data.frame(predictions = c(baseline_predictions$predictions, dnn_predictions$predictions), 
                                            actual = dnn_predictions$actual,
                                            types = prediction_types)

ggplot(prediction_df, aes(x = predictions, y=actual)) +
  xlim(0,11) + ylim(0,11)+
  geom_point(aes(col = types), size = 3)+
  geom_abline(slope = 1, intercept= 0, color = "black", linewidth = 1)+
  annotate('text', x = 2, y = 10, label = paste("Baseline Pearson Coorlation: ", round(baseline_cor,2)), size = 4.3)+
  annotate('text', x = 2.2, y = 9, label = paste("DNN Pearson Coorlation: ", round(dnn_cor,2)), size = 4.3)+
  labs(x = "Predicted Values", y = 'Actual Values', title = "Actual VS Predicted IC50 Values of 5-Fluorouracil", subtitle = "From GDSC2 Data")
  

#bar plot
################################################################################################################################
cor_results = data.frame(models = c("Linear Regression Model", 'Deep Neural Network'),correlations  = c(baseline_cor, dnn_cor))


bar_plot <- ggplot(cor_results, aes(x = models, y = correlations, fill = models )) +
  geom_bar(stat = 'identity') + 
  ylab("Correlation (predicted vs observed)") +
  ggtitle("Model Performance on IC-50 Values of 5-Fluorouracil Treatment")+
  scale_fill_manual(values = c("Linear Regression Model" = "red", 'Deep Neural Network' = "blue"))
print(bar_plot)
