print(getwd())

library(gplots)
library(dplyr)
library(dendextend)
library(ggplot2)


data = read.csv("./data/Training_data.csv",sep=",", header = TRUE,stringsAsFactors = FALSE)

sample_row <- data[1,]


data_modif <- data[, -c(1,2,603)]



# first_200 <- data_modif[,0:200]
# 
# coorlation_matrix <- cor(first_200)
# 
# dend1 <- coorlation_matrix %>% scale %>% dist %>% hclust
# plot(dend1, main  = "")



# dend1 <- coorlation_matrix %>% scale %>% dist %>% hclust %>% as.dendrogram
# ggd1=as.ggdend(dend1)
# 
# ggplot(ggd1, horiz = TRUE, theme = NULL)+
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.text=element_text(color="black"),
#         axis.line = element_line(colour = "black"))
# 



graph <- heatmap.2(coorlation_matrix,
                   main = 'Coorlation Matrix Heatmap Of Genes and IC50 Values',
                   trace = "none",
                   #dendrogram = "both",
                   col = colorRampPalette(c("blue",'white','red'))(100),
                   margins = c(2,2)



                   )
print(graph)
