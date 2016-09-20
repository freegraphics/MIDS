Scripts for preparing MovieLens 1M data set for recommender system and for analysing system work.

I used prepare_data.R to create files in the data directory.

Use show_ids.R and learning_lines.R scripts to watch learning process.
To see ids 
1. Set in mids.py
		train_mode = False
		user_lines_mode = False
	and set index of result folder to watch ids from. For example set  
		indexes = [10]
	to see 10`s result folder ids
2. Run 
		python mids.py 
3. Change directory pathes in the show_ids.R
		items_ids <- read.delim("d:\\works\\projects\\RecommenderSystem\\tests\\items_ids.dta", header=FALSE)
		users_ids <- read.delim("d:\\works\\projects\\RecommenderSystem\\tests\\users_ids.dta", header=FALSE)

To see learning lines 
1. Change directory path in learning_lines.R
		learning <- read.table("d:\\works\\projects\\RecommenderSystem\\tests\\trace.txt", quote="\"", comment.char="")
	to the path of generated trace.txt by the run python mids.py for training
