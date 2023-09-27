library(ggplot2)
library(MASS)
library(ggpubr)

#test data, working to implement actual prediction data
data <- data.frame(
  actual = c(1, 2, 3, 4, 5),
  pred = c(1,2,3,4,6)
)


# density <- kde2d(data$actual, data$y, n=100)
# scatterplot <- ggplot(data, aes(x = x, y = y)) +
#   geom_point() +
#   #geom_density_2d(data = as.data.frame(density), aes(x = x, y = y)) +
#   labs(x = "X", y = "Y", title = "Scatterplot") +
#   theme_minimal()


#code from the paper
calc_density <- function(x, y){
  #KDE calculation
  dens <- kde2d(x, y, n=50)
  
  #which interview from 1 to n each x falls onto
  ix <- findInterval(x, dens$x)
  iy <- findInterval(y, dens$y)
  
  #matrix of x and y intervals
  ii <- cbind(ix, iy)
  
  #KDE values for each (x,y)
  d = dens$z[ii]
  
  return(d)
}
  
data$density<-calc_density(data$actual, data$pred)
  
  


g <- ggscatter(
  data, x = "actual", y = "pred",
  
  #adds a regression line
  add ='reg.line', 
  
  #adds confidence intervals
  conf.int = TRUE, 
  
  #adds coorlation coefficient, p value, change letter size
  cor.coef = TRUE, cor.method = 'pearson', cor.coef.size = 5,
  
  #adds density of each point as a color
  color = "density",
  
  #add coloring
  add.params = list(color="gray"), 
  
  #adds x and y labels
  xlab = "Actual IC50 Value", ylab = 'Predicted IC50 Value', 
  
  
  )
print(g)