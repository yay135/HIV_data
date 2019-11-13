
# to use th model, when we pad the sequences, the zeros need to be all in the left of the sequences.

import torch
import torch.nn as nn


###################################################################

# Weight initial setup

def weight_initial(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            if m.bias is not None:
               nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.uniform_(param, a=-0.01, b=0.01)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            nn.init.constant_(m.bias.data, 0.0)

###################################################################


###################################################################
# feature preprocessing

# suppose the shape of raw input is [batch_size, seq_len, input_channels] 

# need to consider if we put time-variant feature with time-invariant feature together to the fc layers or we just
# put them to separate fc layers and then concat with output.

# mask shape [batch_size, seq_len]
# the output here will be  [batch_size, seq_len, 40]

class Post_encoder(nn.Module):

    def __init__(self, inv_size, v_size):
        super(Post_encoder, self).__init__()
        self.net_inv = nn.Sequential(
            nn.Linear(inv_size, 100),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(50, 20)
        )

        self.net_v = nn.Sequential(
            nn.Linear(v_size, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 20)
        )

        self.out_act = nn.tanh()

    def forward(self, raw_inv,raw_v,mask):
        out_inv = self.net_inv(raw_inv)
        out_inv = self.out_act(out_inv)
        out_v = self.net_v(raw_v)
        out_v = self.out_act(out_v)
        out = torch.cat([out_v, out_inv], dim = 2)
        mask =torch.reshape(mask, (mask.size(0),mask.size(1),1))
        # to mask out the padding positions.
        out = mask * out

        return out


###################################################################


###################################################################

# LSTM_model

class LSTM_main(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size, batch_size,  bias=False):
        super(LSTM_main, self).__init__()

        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.lstm = nn.LSTMCell(self.input_channels, self.hidden_size, bias = bias)     
        # self.act = nn.LeakyReLU()
        self.linear = nn.Linear(self.hidden_size, self.input_channels)
        self.linear2 = nn.Linear(self.input_channels, self.output_size)

        # self.hidden_states = None


    def forward(self, seq_input, future = 0 ):
    	# dimension of seq_input : [batch_size, seq_length, channel_size ]

        intermediates = []
        outputs = []

        #state initialization
        h_t = torch.zeros(seq_input.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(seq_input.size(0), self.hidden_size, dtype=torch.float32)

        # sequence forwarding
        for i, input_t in enumerate(seq_input.chunk(seq_input.size(1), dim=1)):

        	# change the dimension of input_t to [batch_size, channel_size]
        	input_t = input_t.view(input_t.size(0),-1)
            h_t, c_t = self.lstm(input_t, (h_t, c_t))

        	# we will use intermediate as input value to predict future results
            intermediate = self.linear(h_t)
            intermediates += [intermediate]

            output = self.linear2(intermediate) 
            outputs += [output]


        for i in range(future):
            # if y is not None and random.random() > 0.5:
            #     output = y[:, [i]]  # teacher forcing
            h_t, c_t = self.lstm(intermediate, (h_t, c_t))
            intermediate = self.linear(h_t)
            intermediates += [intermediate]
            output = self.linear2(intermediate)      
            outputs += [output]

        outputs = torch.stack(outputs, dim=1)

        return outputs       

###################################################################

###################################################################

# main model

class Model(nn.Module):
    def __init__(self, inv_size, v_size, input_channels, hidden_size, output_size, batch_size):
        super(Model, self).__init__()

        self.feature_compress = Post_encoder(inv_size, v_size)
        self.lstm_predictor = LSTM_main(input_channels, hidden_size, output_size, batch_size)

    def forward( self, raw_inv,raw_v,mask, self, seq_input, future ):

        post_seq = self.feature_compress( raw_inv, raw_v, mask )
        predict_results = self.lstm_predictor (seq_input, future)

        return predict_results


