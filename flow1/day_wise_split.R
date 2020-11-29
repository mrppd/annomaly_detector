file_name = paste("1/1_reduced", sep = "")
df_auth_chunk_reduced = read.csv(file_name, header = F, sep = ",")

for(i in c(2:59)){
  file_name = paste("1/", as.character(i), "_reduced", sep = "")
  print(file_name)
  df_auth_chunk_reduced2 = read.csv(file_name, header = F, sep = ",")
  
  df_auth_chunk_reduced = rbind(df_auth_chunk_reduced, df_auth_chunk_reduced2)

}

for(i in c(10:10)){
  day_no=i
  day_start = (day_no-1)*(24*60*60)+1
  day_end = day_no*(24*60*60)+1
  cat(day_no, day_start, day_end, "\n")
  auth_day = subset(df_auth_chunk_reduced, (V1>=day_start & V1<day_end) )
  write.table(auth_day, file=paste("days/day", as.character(day_no), "_reduced", sep = ""), sep=",", row.names = F, col.names=F)
}


df_auth_chunk_reduced = subset(df_auth_chunk_reduced, V1>=(10*(24*60*60)) )


for(i in c(1:59)){
  file_name = paste("2/", as.character(i), "_reduced", sep = "")
  print(file_name)
  df_auth_chunk_reduced2 = read.csv(file_name, header = F, sep = ",")
  
  df_auth_chunk_reduced = rbind(df_auth_chunk_reduced, df_auth_chunk_reduced2)
}

for(i in c(11:20)){
  day_no=i
  day_start = (day_no-1)*(24*60*60)+1
  day_end = day_no*(24*60*60)+1
  cat(day_no, day_start, day_end, "\n")
  auth_day = subset(df_auth_chunk_reduced, (V1>=day_start & V1<day_end) )
  print(nrow(auth_day))
  write.table(auth_day, file=paste("days/day", as.character(day_no), "_reduced", sep = ""), sep=",", row.names = F, col.names=F)
}



