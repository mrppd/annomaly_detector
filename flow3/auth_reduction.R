#!/usr/bin/env Rscript
argv = commandArgs(trailingOnly=TRUE)
folder_name = ""#argv[1]
file_name = "Test_day_9_10000N200AN.txt"#argv[2]
file = paste(folder_name, file_name, sep = "")

df_auth_chunk = read.csv(file, header = T, sep = ",")
df_auth_chunk = within(df_auth_chunk, rm(X))
names(df_auth_chunk) = c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10")

df_auth_chunk_compromised = subset(df_auth_chunk, V10==1) 
write.table(df_auth_chunk_compromised, file=paste(file,"_compromised", sep = ""), sep=",", row.names = F, col.names=F)


df_auth_chunk$V4 = as.character(df_auth_chunk$V4)
df_auth_chunk$V5 = as.character(df_auth_chunk$V5)
df_auth_chunk$V9 = as.character(df_auth_chunk$V9)

df_auth_chunk_reduced = subset(df_auth_chunk, V4!=V5)

df_auth_chunk_reduced = within(df_auth_chunk_reduced, rm(V2, V3, V8, V6, V7))
df_auth_chunk_reduced$V9[df_auth_chunk_reduced$V9=="Success"]=1
df_auth_chunk_reduced$V9[df_auth_chunk_reduced$V9=="Fail"]=0

write.table(df_auth_chunk_reduced, file=paste(file,"_reduced", sep = ""), sep=",", row.names = F, col.names=F)
cat("File reduction successful for file:", file, "\n")
