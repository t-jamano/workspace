from Models import *

class Seq2Seq():
    
    def __init__(self, nb_words, max_len, embedding_matrix, dim, optimizer=Adam(), keep_rate_word_dropout=0.75):

        self.dim = dim
        self.nb_words = nb_words
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.keep_rate_word_dropout = keep_rate_word_dropout

        self.hidden_dim = self.dim[0]
        self.latent_dim = self.dim[1]

        self.build()

    def build(self):

        query_inputs = Input(shape=(self.max_len,))
        doc_inputs = Input(shape=(self.max_len,))
        label_inputs = Input(shape=(1,))
        kl_inputs = Input(shape=(1,))

        encoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)

        norm = BatchNormalization()


        encoder_lstm = GRU(self.hidden_dim)


        state = norm(encoder_lstm(encoder_embedding(query_inputs)))
        self.mean = Dense(self.latent_dim)

        state = self.mean(state)
        state_z = state


        decoder_inputs = Input(shape=(self.max_len,))

        self.latent2hidden = Dense(self.hidden_dim)
        self.decoder_lstm = GRU(self.hidden_dim, return_sequences=True, name="dec_gru")
        self.decoder_dense = Dense(self.nb_words, activation='softmax', name="rec")
        self.decoder_embedding = Embedding(self.nb_words,
                                        self.embedding_matrix.shape[-1],
                                        weights=[self.embedding_matrix],
                                        input_length=self.max_len,
                                        mask_zero=True,
                                        trainable=True)
        
        rec_outputs = self.decoder_dense(self.decoder_lstm(self.decoder_embedding(decoder_inputs) , initial_state=self.latent2hidden(state_z)))
        inputs = [query_inputs, decoder_inputs]
        self.model = Model(inputs, [rec_outputs])
        self.model.compile(optimizer=self.optimizer, loss="sparse_categorical_crossentropy")        
                
        self.encoder = Model(query_inputs, state)

    def name(self):
    	return "s2s" 
    
    def word_dropout(self, x, unk_token):

        x_ = np.copy(x)
        rows, cols = np.nonzero(x_)
        for r, c in zip(rows, cols):
            if random.random() <= self.keep_rate_word_dropout:
                continue
            x_[r][c] = unk_token

        return x_
