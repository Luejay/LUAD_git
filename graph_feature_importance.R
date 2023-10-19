library(ggplot2)
library(dplyr)
library(dendextend)

data <- read.csv("./feature_importance/dnn/2023_10_17_19_34_06.csv")
colnames(data)[colnames(data) == "permutation"] <- "Z_Score"

data_filtered <- dplyr::arrange(data,desc(Z_Score))[1:20,]

data_filtered$feature_name <- reorder(data_filtered$feature_name, data_filtered$Z_Score)

ggplot(data_filtered, aes(x = 0, y = feature_name, fill = Z_Score)) +
        coord_fixed(ratio = 1)+
        geom_tile() +
        theme_minimal() +
        labs(x = NULL, y = "Genes", title = "DNN Most Important Gene Feature in 5-Fluorouracil Prediction")+
        theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
        scale_fill_gradient2(low = 'blue', mid = 'white',high = 'red', midpoint = 0) #+




expression_df <- read.csv("./data/Training_data.csv")

expression_df_filter <- expression_df[, colnames(expression_df) %in% data_filtered$feature_name]

matrix <- cor(expression_df_filter)
clusters <- matrix %>% scale %>% dist %>% hclust

dend <- as.dendrogram(clusters)

ggplot(dend, horiz = TRUE, theme = NULL)+ 
  #remove major and minor grids, background color, axis text to be black, color of line to be black
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.text=element_text(color="black"), 
        axis.line = element_line(colour = "black"))
  