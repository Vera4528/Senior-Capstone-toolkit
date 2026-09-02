no1 <- function (v, p){
  x <- matrix(0, nrow = v, ncol = p)
  for(i in 1:v){
    z <- rnorm(p)
    zsum <- 0
    zsq <- z^2
    x[i, ] <- z / sqrt(sum(zsq))
  }
  
  return(x)
}