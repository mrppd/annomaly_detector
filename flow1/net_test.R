#!/usr/bin/env Rscript
argv = commandArgs(trailingOnly=TRUE)

file_name = argv[1]
cat("Started processing file named: ", file_name, "\n", sep = "")
df = read.table(file_name, header = FALSE, sep = ",")

count = rep(0L, 5011199)

for(i in c(1: nrow(df))){
  count[df$V1[i]] = count[df$V1[i]]+1
}

df_histo = data.frame(time=c(1:5011199), hist=count)

cat("Writing partial result: ", paste(file_name,".csv", sep = ""), "\n", sep = "")
write.csv(df_histo, file=paste(file_name,".csv", sep = ""), row.names = F)

