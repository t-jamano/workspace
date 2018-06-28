def qq_batch_generator(reader, tokeniser, batch_size, max_len, nb_words):
	while True:
		for df in reader:
		    q = df.q.tolist()
		    if train_data == "1M_EN_QQ_log2":
		    	d = [i.split("<sep>")[0] for i in df.d.tolist()]
		    else:
		    	d = df.d.tolist()
		    
		    q = pad_sequences(tokeniser.texts_to_sequences(q), maxlen=max_len)
		    d = pad_sequences(tokeniser.texts_to_sequences(d), maxlen=max_len)
		    
		    q_one_hot = np.zeros((batch_size, nb_words))
		    for i in range(len(q)):
		        q_one_hot[i][q[i]] = 1
		        
		    d_one_hot = np.zeros((batch_size, nb_words))
		    for i in range(len(d)):
		        d_one_hot[i][d[i]] = 1
		        
		        
		    # negative sampling from positive pool
		    neg_d_one_hot = [[] for j in range(J)]
		    for i in range(batch_size):
		        possibilities = list(range(batch_size))
		        possibilities.remove(i)
		        negatives = np.random.choice(possibilities, J, replace = False)
		        for j in range(J):
		            negative = negatives[j]
		            neg_d_one_hot[j].append(d_one_hot[negative].tolist())
		    
		    y = np.zeros((batch_size, J + 1))
		    y[:, 0] = 1
		    
		    for j in range(J):
		        neg_d_one_hot[j] = np.array(neg_d_one_hot[j])
		    
		#         print(q_one_hot.shape, d_one_hot.shape, len(neg_d_one_hot))
		#         print(neg_d_one_hot[0])

		    # negative sampling from randomness
		    # for j in range(J):
		    #     neg_d_one_hot[j] = np.random.randint(2, size=(batch_size, 10, WORD_DEPTH))
		    

		#         q_one_hot = to_categorical(q, nb_words)   
		#         q_one_hot = q_one_hot.reshape(batch_size, max_len, nb_words)
		    
		    
		    yield [q_one_hot, d_one_hot] + [neg_d_one_hot[j] for j in range(J)], y


