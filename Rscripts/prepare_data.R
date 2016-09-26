#
# the script to prepare data for the recommender system
# 

# first convert MovieLens 1M dataset files to the cvs file format

MovieLens1MPath <- "d:\\works\\github\\mids\\ml-1m\\"
ResultCSVPath <- "d:\\works\\github\\mids\\data\\"

MovieLens1MMovieFileName <- paste(MovieLens1MPath,"movies.dat", sep="")
movies1m <- read.delim(MovieLens1MMovieFileName,header=FALSE, quote="",stringsAsFactors=FALSE)
movies1mSp <- strsplit(movies1m$V1,"::",perl=TRUE)

movies_csv <- data.frame(matrix(nrow = length(movies1mSp),ncol = 3))
for(i in 1:length(movies1mSp))
{
  movies_csv[i,1] <- as.integer(movies1mSp[[i]][1])
  movies_csv[i,2] <- movies1mSp[[i]][2]
  movies_csv[i,3] <- movies1mSp[[i]][3]
}
MoviesCSVFileName <- paste(ResultCSVPath,"pre_movies.csv")
write.table(movies_csv,file = MoviesCSVFileName, row.names = FALSE,  col.names=FALSE, sep=";", quote = TRUE)

MovieLens1MUsersFileName <- paste(MovieLens1MPath,"users.dat",sep="")
users1m <- read.delim(MovieLens1MUsersFileName,header=FALSE, quote="",stringsAsFactors=FALSE)
users1mSp <- strsplit(users1m$V1,"::",perl=TRUE)

users_csv <- data.frame(matrix(nrow = length(users1mSp),ncol = 5))
for(i in 1:length(users1mSp))
{
  users_csv[i,1] <- as.integer(users1mSp[[i]][1])
  users_csv[i,2] <- users1mSp[[i]][2]
  users_csv[i,3] <- users1mSp[[i]][3]
  users_csv[i,4] <- users1mSp[[i]][4]
  users_csv[i,5] <- users1mSp[[i]][5]
}

UsersCSVFileName <- paste(ResultCSVPath,"pre_users.csv")
write.table(users_csv,file = UsersCSVFileName, row.names = FALSE, col.names=FALSE, sep=";", quote = TRUE)

MovieLens1MRatingsFileName <- paste(MovieLens1MPath,"ratings.dat",sep = "")
ratings1m <- read.delim(MovieLens1MRatingsFileName,header=FALSE, quote="",stringsAsFactors=FALSE)
ratings1mSp <- strsplit(ratings1m$V1,"::",perl=TRUE)

ratings_csv <- data.frame(matrix(nrow = length(ratings1mSp),ncol = 4))
for(i in 1:length(ratings1mSp))
{
  ratings_csv[i,1] <- ratings1mSp[[i]][1]
  ratings_csv[i,2] <- ratings1mSp[[i]][2]
  ratings_csv[i,3] <- ratings1mSp[[i]][3]
  ratings_csv[i,4] <- ratings1mSp[[i]][4]
}

RatingsCSVFileName <- paste(ResultCSVPath,"pre_ratings.csv")
write.table(ratings_csv,file = RatingsCSVFileName, row.names = FALSE, col.names=FALSE, sep=";", quote = TRUE)

