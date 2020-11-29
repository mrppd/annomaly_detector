df_sec = read.csv("res_combined.csv", header = T, sep = ",")

plot(df_sec$hist, xlab = "Time (s)", ylab = "Count")

day=8
tmp = df_sec$hist[(day*24*60*60):((day*24*60*60)+(24*60*60))]
plot(c((day*24*60*60):((day*24*60*60)+(24*60*60))), tmp, xlab = "Time (s)", ylab = "Count", type = 'l')


day=9
tmp = df_sec$hist[(day*24*60*60):((day*24*60*60)+(24*60*60))]
plot(c((day*24*60*60):((day*24*60*60)+(24*60*60))), tmp, xlab = "Time (s)", ylab = "Count", type = 'l')