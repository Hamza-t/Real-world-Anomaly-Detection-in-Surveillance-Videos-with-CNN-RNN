import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length

        self.lstm = nn.LSTM(1, input_size, hidden_size,
                            num_layers)  # lstm
        self.fc_1 = nn.Linear(512, 32)  # fully connected 1
        self.fc = nn.Linear(32, 1)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(1, x.size(0), 512))  # hidden state
        c_0 = Variable(torch.zeros(1, x.size(0), 512))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out

    '''
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2)
        self.fc1 = nn.Linear(in_features=512, out_features=32)
        self.fc2 = nn.Linear(32, 1)


    def forward(self, x):
       
        h = torch.zeros((2, x.size(0), 512))
        c = torch.zeros((2, x.size(0), 512)) #andi mochkle lena

        #h = torch.randn(2, 0, 512)
        #c = torch.randn(2, 0, 512)
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)


        out, (hidden, cell) = self.lstm(x, (h, c))
        out = self.dropout(out)
        out = torch.relu_(self.fc1(out[:, -1]))
        out = self.dropout(out)
        out = out.view(-1, 2)
        out = torch.log_softmax(self.fc2(out))
        
        h_0 = torch.zeros(1, x.size(0), 512)  # hidden state
        c_0 = torch.zeros(1, x.size(0), 512)  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


    
    def forward(self, x):
        output, _status = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc1(torch.relu(output))
        return output
    '''
