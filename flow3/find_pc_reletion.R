#!/usr/bin/env Rscript
argv = commandArgs(trailingOnly=TRUE)
argv[1] = "Test_day_9_10000N200AN.txt_reduced"
path = argv[1] #paste("days/day", as.character(argv[1]), "_reduced", sep = "")
df_auth_chunk_reduced = read.csv(path, header = F, sep = ",")
df_auth_chunk_reduced$id = c(1:nrow(df_auth_chunk_reduced))

C = paste(df_auth_chunk_reduced$V2, " ", df_auth_chunk_reduced$V3, sep = "")
C_unique = unique(C)

df_auth_chunk_reduced$V2 = as.character(df_auth_chunk_reduced$V2)
df_auth_chunk_reduced$V3 = as.character(df_auth_chunk_reduced$V3)
df_auth_chunk_reduced$V4 = as.integer(df_auth_chunk_reduced$V4)


src_des = data.frame("src", "des", "times", "failed", "id_list", stringsAsFactors=FALSE)
names(src_des) = c("src", "des", "times", "failed", "id_list")

for(j in c(1:length(C_unique))){
  src = unlist(strsplit(C_unique[j], " "))[1]
  des = unlist(strsplit(C_unique[j], " "))[2] 
  sub_chunk = subset(df_auth_chunk_reduced, (V2==src & V3==des))
  id_list = paste(sub_chunk$id, sep="", collapse =" ")
  sub_chunk_failed = subset(sub_chunk, V4==0)
  
  #cat(src, des, "\n", sep = " ")
  #print(nrow(sub_chunk_failed))
  
  #cat(src, des, nrow(sub_chunk), nrow(sub_chunk_failed), "\n", sep = "\t")  
  
  src_des[nrow(src_des) + 1,] = list(src, des, nrow(sub_chunk), nrow(sub_chunk_failed), id_list)
  if(j%%1000==0){
    print(j)
  }
}
src_des = src_des[2:nrow(src_des),]

write.csv(src_des, file=paste("src_des", as.character(argv[1]), ".csv", sep=""), row.names = F)



# 
# pc_count = vector(mode="numeric", length=length(C_unique))
# names(pc_count) = C_unique
# for(i in c(1:length(C_unique))){
#   pc_count[as.character(C_unique[i])] = 0
# }
# 
# for(i in c(1:length(C))){
#   pc_count[as.character(C[i])] = as.integer(pc_count[as.character(C[i])]) + 1
#   if(i%%100000==0)
#     print(i)
# }
# 
# 
# summary(pc_count)
# pc_count[as.character(C_unique[100])]



