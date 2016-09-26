#
# the script to draw ids of the recommender system run
#

# [d:\\works\\projects\\RecommenderSystem\\tests\\] -- the folder where we run recommender system
items_ids <- read.delim("F:\\works\\projects\\python\\RecommenderSystem\\tests\\16.09.24 -- 17-35\\ids_005\\items_ids.dta", header=FALSE)
users_ids <- read.delim("F:\\works\\projects\\python\\RecommenderSystem\\tests\\16.09.24 -- 17-35\\ids_005\\users_ids.dta", header=FALSE)

par(col="black")
plot(items_ids_01$V2,items_ids_01$V3,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(items_ids$V2,items_ids$V3,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(items_ids_01$V4,items_ids_01$V5,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(items_ids$V4,items_ids$V5,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(items_ids_01$V6,items_ids_01$V7,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(items_ids$V6,items_ids$V7,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(items_ids_01$V8,items_ids_01$V9,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(items_ids$V8,items_ids$V9,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(items_ids_01$V10,items_ids_01$V11,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(items_ids$V10,items_ids$V11,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(items_ids_01$V12,items_ids_01$V13,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(items_ids$V12,items_ids$V13,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(items_ids_01$V14,items_ids_01$V15,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(items_ids$V14,items_ids$V15,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids_01$V2,users_ids_01$V3,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids$V2,users_ids$V3,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids_01$V4,users_ids_01$V5,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids$V4,users_ids$V5,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids_01$V6,users_ids_01$V7,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids$V6,users_ids$V7,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids_01$V8,users_ids_01$V9,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids$V8,users_ids$V9,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids_01$V10,users_ids_01$V11,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids$V10,users_ids$V11,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids_01$V12,users_ids_01$V13,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids$V12,users_ids$V13,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids_01$V14,users_ids_01$V15,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))
plot(users_ids$V14,users_ids$V15,xlim = c(-0.5,0.5),ylim = c(-0.5,0.5))

hist(items_ids$V2,breaks = 35)  
hist(items_ids$V3,breaks = 35)  
hist(items_ids$V4,breaks = 35)  
hist(items_ids$V5,breaks = 35)  
hist(items_ids$V6,breaks = 35)  
hist(items_ids$V7,breaks = 35)  
hist(items_ids$V8,breaks = 35)  
hist(items_ids$V9,breaks = 35)  
hist(items_ids$V10,breaks = 35)  
hist(items_ids$V11,breaks = 35)  
hist(items_ids$V12,breaks = 35)  
hist(items_ids$V13,breaks = 35)  
hist(items_ids$V14,breaks = 35)  
hist(items_ids$V15,breaks = 35)  

# update ids for the next step
items_ids_01 <- items_ids
users_ids_01 <- users_ids
