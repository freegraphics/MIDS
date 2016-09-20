#
# the script to prepare data for the recommender system
# 

# first convert MovieLens 1M dataset files to the cvs file format

MovieLens1MPath <- "d:\\works\\github\\mids\\ml-1m\\"
ResultCSVPath <- "d:\\works\\github\\mids\\data\\"

MovieLens1MMovieFileName <- paste(MovieLens1MPath,"movies.dat", sep="")
movies1m <- read.delim(MovieLens1MMovieFileName,header=FALSE, quote="",stringsAsFactors=FALSE)
movies1mSp <- strsplit(movies1m$V1,"::",perl=TRUE)

ids <- c()
names <- c()
genders <- c()
for(i in 1:length(movies1mSp))
{
  ids <- c(ids,as.integer(movies1mSp[[i]][1]))
  names <- c(names,movies1mSp[[i]][2])
  genders <- c(genders,movies1mSp[[i]][3])
}
movies_csv <- data.frame(ids,names,genders)
MoviesCSVFileName <- paste(ResultCSVPath,"pre_movies.csv")
write.table(movies_csv,file = MoviesCSVFileName, row.names = FALSE,  col.names=FALSE, sep=";", quote = TRUE)

MovieLens1MUsersFileName <- paste(MovieLens1MPath,"users.dat",sep="")
users1m <- read.delim(MovieLens1MUsersFileName,header=FALSE, quote="",stringsAsFactors=FALSE)
users1mSp <- strsplit(users1m$V1,"::",perl=TRUE)

ids <- c()
genders <- c()
ages <- c()
accupations <- c()
zipcodes <- c()
for(i in 1:length(users1mSp))
{
  ids <- c(ids,as.integer(users1mSp[[i]][1]))
  genders <- c(genders,users1mSp[[i]][2])
  ages <- c(ages,users1mSp[[i]][3])
  accupations <- c(accupations,users1mSp[[i]][4])
  zipcodes <- c(zipcodes,users1mSp[[i]][5])
}

users_csv <- data.frame(ids,genders,ages,accupations,zipcodes)
UsersCSVFileName <- paste(ResultCSVPath,"pre_users.csv")
write.table(users_csv,file = UsersCSVFileName, row.names = FALSE, col.names=FALSE, sep=";", quote = TRUE)

MovieLens1MRatingsFileName <- paste(MovieLens1MPath,"ratings.dat",sep = "")
ratings1m <- read.delim(MovieLens1MRatingsFileName,header=FALSE, quote="",stringsAsFactors=FALSE)
ratings1mSp <- strsplit(ratings1m$V1,"::",perl=TRUE)

user_ids <- c()
movie_ids <- c()
ratings <- c()
timestamps <- c()
for(i in 1:length(ratings1mSp))
{
  user_ids <- c(user_ids,ratings1mSp[[i]][1])
  movie_ids <- c(movie_ids,ratings1mSp[[i]][2])
  ratings <- c(ratings,ratings1mSp[[i]][3])
  timestamps <- c(timestamps,ratings1mSp[[i]][4])
}

ratings_csv <- data.frame(user_ids,movie_ids,ratings,timestamps)
RatingsCSVFileName <- paste(ResultCSVPath,"pre_ratings.csv")
write.table(ratings_csv,file = RatingsCSVFileName, row.names = FALSE, col.names=FALSE, sep=";", quote = TRUE)
