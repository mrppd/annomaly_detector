#!/usr/bin/env Rscript
argv = commandArgs(trailingOnly=TRUE)
#argv[1] = 9
path = paste("src_des_v2/src_des", as.character(argv[1]), ".csv", sep = "")
src_des = read.csv(path, header = T, sep = ",")
src_des$src = as.character(src_des$src)
src_des$des = as.character(src_des$des)
src_des$times = as.integer(src_des$times)
src_des$failed = as.integer(src_des$failed)

src_unique = as.character(unique(src_des$src))


src_info = data.frame("src", "total_con", "num_des", "total_failed", "des_list", "id_list", stringsAsFactors=FALSE)
names(src_info) = c("src", "total_con", "num_of_des", "total_failed", "des_list", "id_list")
cat("Total Iter: ", length(src_unique), "\n")

for(i in c(1:length(src_unique))){
  src_des_sub = subset(src_des, src==src_unique[i])
  des_list = paste(src_des_sub$des, collapse = " ")
  id_list = paste(src_des_sub$id_list, collapse = " ")
  #cat(i, src_unique[i], nrow(src_des_sub), sum(src_des_sub$times), sum(src_des_sub$failed), "\n")
  
  src_info[nrow(src_info) + 1,] = list(src_unique[i], sum(src_des_sub$times), nrow(src_des_sub), sum(src_des_sub$failed), as.character(des_list), as.character(id_list))
  if(i%%2000==0){
    print(i)
  }
}

src_info = src_info[2:nrow(src_info),]
src_info$total_con = as.numeric(src_info$total_con)
src_info$num_of_des = as.numeric(src_info$num_of_des)
src_info$total_failed = as.numeric(src_info$total_failed)

src_info$con_per_des = src_info$num_of_des/src_info$total_con
src_info$success_to_fail_ratio = src_info$total_failed/(src_info$total_con - src_info$total_failed)

src_info_order_by_num_des = src_info[order(src_info$num_of_des, src_info$total_failed, -src_info$con_per_des, decreasing=T),] 

write.csv(src_info_order_by_num_des, file=paste("src_info_v2/src_info", as.character(argv[1]), ".csv", sep=""), row.names = F)

cat("File src_info", as.character(argv[1]), " has been created successfully.", "\n", sep="")