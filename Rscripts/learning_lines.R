#
# the script to draw learning lines 
#

# [F:\\works\\projects\\python\\RecommenderSystem\\tests\\16.09.24 -- 17-35\\trace.txt] -- the folder where we run recommender system

learning <- read.table("F:\\works\\projects\\python\\RecommenderSystem\\tests\\16.09.24 -- 17-35\\trace.txt", quote="\"", comment.char="")
x <- c(min(learning$V1),max(learning$V1))
y <- c(0,max(learning$V2))
par(col="black")
plot(x,y)
lines(learning$V1,learning$V2)
lines(learning$V1,learning$V3)
lines(learning$V1,learning$V4)
lines(learning$V1,learning$V5)
lines(learning$V1,learning$V6)
lines(learning$V1,learning$V7)
#lines(learning$V1,learning$V9)
y <- c(min(learning$V2),min(learning$V2))
lines(x,y)
y <- c(min(learning$V3),min(learning$V3))
lines(x,y)

