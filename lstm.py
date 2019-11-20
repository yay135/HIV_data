
# to use th model, when we pad the sequences, the zeros need to be all in the left of the sequences.
# for this version, we only need to predict next periods of next two visits 

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

# class Post_encoder(nn.Module):

#     def __init__(self, inv_size, v_size):
#         super(Post_encoder, self).__init__()
#         self.net_inv = nn.Sequential(
#             nn.Linear(inv_size, 100),
#             nn.LeakyReLU(),
#             nn.Dropout(0.5),

#             nn.Linear(100, 50),
#             nn.LeakyReLU(),
#             nn.Dropout(0.5),

#             nn.Linear(50, 20)
#         )

#         self.net_v = nn.Sequential(
#             nn.Linear(v_size, 20),
#             nn.LeakyReLU(),
#             nn.Linear(20, 20)
#         )

#         self.out_act = nn.tanh()

#     def forward(self, raw_inv,raw_v,mask):
#         out_inv = self.net_inv(raw_inv)
#         out_inv = self.out_act(out_inv)
#         out_v = self.net_v(raw_v)
#         out_v = self.out_act(out_v)
#         out = torch.cat([out_v, out_inv], dim = 2)
#         mask =torch.reshape(mask, (mask.size(0),mask.size(1),1))
#         # to mask out the padding positions.
#         out = mask * out

#         return out


###################################################################


###################################################################

# LSTM_model

class LSTM_main(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size, batch_size, layers,  bias=False):
        super(LSTM_main, self).__init__()
        self.lstm = nn.LSTM(
            input_channels,
            hidden_size,
            num_layers=layers,
            bias=bias,
            batch_first=True,
            bidirectional=False,
            dropout=0.5
        )
        self.layers = layers
        self.num_directions = 1
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden_states = None

        self.act = nn.LeakyReLU()

        self.linear = nn.Linear(self.hidden_size, self.output_size)


    def init_hidden_state(self):
        return (
            torch.zeros(self.layers * self.num_directions,
                        self.batch_size, self.hidden_size),
            torch.zeros(self.layers * self.num_directions,
                        self.batch_size, self.hidden_size)
        )         


    def forward(self, seq_input):
    	# dimension of seq_input : [batch_size, seq_length, channel_size ]

        self.hidden_states = self.init_hidden_state()

        out, self.hidden_states = self.lstm(seq_input, self.hidden_states)

        # This will ouput the hidden state of the last layer at last timestamp. The output size will be [batch_size, hidden_size]
        temp = self.hidden_states[0][self.layers-1]

        out = self.act(temp)

        # this will output a vector of [batch_size, 2]
        out = self.linear(out)

        return out     

###################################################################

###################################################################

# main model

class Model(nn.Module):
    def __init__(self, input_channels=1, hidden_size=5, output_size=2, batch_size=16, layers=2):
        super(Model, self).__init__()

        # self.feature_compress = Post_encoder(inv_size, v_size)
        self.lstm_predictor = LSTM_main(input_channels, hidden_size, output_size, batch_size,layers)

    def forward( self, seq_input):

        # post_seq = self.feature_compress( raw_inv, raw_v, mask )
        predict_results = self.lstm_predictor (seq_input)

        return predict_results


