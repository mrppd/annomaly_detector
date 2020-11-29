df = read.csv("res_combined.csv", header = T, sep = ",")


#barplot(df$hist, xlab="Time (s)", ylab="Service Count", main="Service count per second (auth data)")

df$time_min = as.integer(df$time/60)+1
df_min = data.frame("time_min", "count", "sd", "var", "mean", "min", "max", "sec_from", "sec_to", stringsAsFactors=FALSE)
names(df_min) = c("time_min", "count", "sd", "var", "mean", "min", "max", "sec_from", "sec_to")

print(as.integer(nrow(df)/60)+1)

for(i in c(1:(as.integer(nrow(df)/60)+1))){
  newdata <- subset(df, time_min==i)
  if(nrow(newdata)>0){
    count_min = sum(newdata$hist)
    sd_min = sd(newdata$hist)
    mean_min = mean(newdata$hist)
    max_min = max(newdata$hist)
    min_min = min(newdata$hist)
    variance_min = var(newdata$hist)
    start_second = min(newdata$time)
    end_second = max(newdata$time)
    
    
    df_min[nrow(df_min) + 1,] = list(i, count_min, sd_min, variance_min, mean_min, min_min, max_min, start_second, end_second)
  }
  
  if(i%%300==0)
    cat(i, " ", sep = "")
}

df_min = df_min[2:nrow(df_min),]
write.csv(df_min, file="service_per_min2.csv", row.names = F)
