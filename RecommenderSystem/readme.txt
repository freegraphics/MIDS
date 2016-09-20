-------------------------------------------------
The file Consts.py contains class Consts of the constants for the recommender system demo.

-------------------------------------------------
The file Nets.py contains simple classes for the neural net and the autoencoder net.

-------------------------------------------------
The file MIDS.py contains demo for base functions of the recommender system.

1. To prepare data for the recommender system demo uncomment the call to the function prepare_data() in the module run if statement. 

2. To train the recommender system demo set 
		train_mode = True
	in the module run if statement.
	and run
		python mids.py
		
3. To get nearest movies and convert npy ids files to the text format set 
		train_mode = False
		user_lines_mode = False
	and run 
		python mids.py
	(may be your should set movie id(s) in function nearest_movies() at line 
		movie_ids = [1251] #,1974
	to see nearest movies for the specific movie ids)
