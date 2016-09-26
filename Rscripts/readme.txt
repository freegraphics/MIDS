Scripts for preparing MovieLens 1M data set for recommender system and for analysing system work.

I used prepare_data.R to create files in the data directory.

Use show_ids.R and learning_lines.R scripts to watch learning process.
To see ids 
1. Set in mids.py
		train_mode = False
		get_best_films_for_users_mode = False
	and set index of result folder to watch ids from. For example set  
		indexes = [10]
	to see 10`s result folder ids
2. Run 
		python mids.py 
3. Change directory pathes in the show_ids.R
		items_ids <- read.delim("F:\\works\\projects\\python\\RecommenderSystem\\tests\\16.09.24 -- 17-35\\ids_005\\items_ids.dta", header=FALSE)
		users_ids <- read.delim("F:\\works\\projects\\python\\RecommenderSystem\\tests\\16.09.24 -- 17-35\\ids_005\\users_ids.dta", header=FALSE)

To see learning lines 
1. Change directory path in learning_lines.R
		learning <- read.table("F:\\works\\projects\\python\\RecommenderSystem\\tests\\16.09.24 -- 17-35\\trace.txt", quote="\"", comment.char="")
	to the path of generated trace.txt by the run python mids.py for training
