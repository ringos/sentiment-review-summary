import torch
import torch.nn as nn



# def mean_pooling(batch,  lengths):
#     '''
#         avg-pooling for pytorch 1.1.0 or beyond,  where pack_padded_sequence is available
#     '''
#     return (torch.sum(batch,  dim=1).transpose(0, 1)/torch.tensor(lengths,  dtype=torch.float).to('cuda')).transpose(0, 1)


'''
pytorch 1.0.1 avg-pooling
'''
def mean_pooling(batch, lengths):
    '''
     - to apply mean_pooling for RNN-modules
     - input:
            batch: [batch_size, seq_len, hid_size]
            lengths: the true length of each sequence
     - output: [batch_size, hid_size]
    '''
    for i, _ in enumerate(batch):
        if int(lengths[i])==0:
            lengths[i]+=1
        if i==0:
            tmp_vec=nn.functional.avg_pool1d(_.transpose(0, 1).unsqueeze(0), int(lengths[i]))[:, :, 0]
        else:
            tmp_vec=torch.cat([tmp_vec, nn.functional.avg_pool1d(_.transpose(0, 1).unsqueeze(0), int(lengths[i]))[:, :, 0]])
    return tmp_vec

class LayerNorm(nn.Module):
    def __init__(self,  hidden_size,  eps=1e-12):
        '''
        Construct a layernorm module in the TF style (epsilon inside the square root).
        '''
        super(LayerNorm,  self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self,  x):
        u = x.mean(-1,  keepdim=True)
        s = (x - u).pow(2).mean(-1,  keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BiLSTM_centric_layer(nn.Module):
    '''
    basic BiLSTM review-centric layer
    input:
        in_size - input_size for LSTM
        hidden_size - hidden_size for LSTM
        in_raw - raw_text,  [batch_size,  seq_len,  vec_dim]
        in_sum - summary_text,  [batch_size,  seq_len,  vec_dim]
        len_raw/len_sum - list of original (unpadded) lengths of the current batch of raw/summary text
    '''
    def __init__(self, in_size, hidden_size, num_heads=4, dropout_rate=0.2):
        super(BiLSTM_centric_layer, self).__init__()
        self.in_size=in_size
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.dropout_rate=dropout_rate
        self.dropout_layer=nn.Dropout(p=self.dropout_rate)
        self.lstm_raw=nn.LSTM(self.in_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.lstm_sum=nn.LSTM(self.in_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.proj_layers_query=nn.ModuleList([nn.Linear(self.hidden_size*2, int(self.hidden_size*2/self.num_heads), bias=False) for i in range(self.num_heads)])
        self.proj_layers_key=nn.ModuleList([nn.Linear(self.hidden_size*2, int(self.hidden_size*2/self.num_heads), bias=False) for i in range(self.num_heads)])
        self.proj_layers_value=nn.ModuleList([nn.Linear(self.hidden_size*2, int(self.hidden_size*2/self.num_heads), bias=False) for i in range(self.num_heads)])
        # self.projection_layer=nn.Linear(self.hidden_size*4*self.num_heads, self.hidden_size*4)
    def forward(self, 
                in_raw, 
                in_sum, 
                len_raw, 
                len_sum, 
                use_sum_lstm=False, 
                output_sum=None, 
                use_residual=True,  
                use_sum_avg_pooling=True, 
                use_concate_raw=False, 
                use_concate_sum=False, 
                use_divide_dk=False):
        output_raw, (h_n_raw, c_n_raw)=self.lstm_raw(in_raw)
        output_raw=self.dropout_layer(output_raw)
        if use_sum_lstm:
            output_sum, (h_n_sum, c_n_sum)=self.lstm_sum(in_sum)
            output_sum=self.dropout_layer(output_sum)
        sum_vec=mean_pooling(output_sum, len_sum)
        sum_vec=self.dropout_layer(sum_vec)
        rst=[]
        for j in range(self.num_heads):
            if use_sum_avg_pooling:
                tmp_query=self.proj_layers_query[j](output_raw)
                tmp_key=self.proj_layers_key[j](sum_vec)
                tmp_value=self.proj_layers_value[j](sum_vec)
                if use_divide_dk:
                    tmp_rst=torch.bmm((nn.Softmax(dim=-1)(torch.bmm(tmp_query, tmp_key.unsqueeze(-1)).squeeze(-1)/((self.hidden_size*2/self.num_heads)**0.5))).unsqueeze(-1), tmp_value.unsqueeze(-2))
                else:
                    tmp_rst=torch.bmm(nn.Softmax(dim=-1)(torch.bmm(tmp_query, tmp_key.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1), tmp_value.unsqueeze(-2))
            else:
                tmp_query=self.proj_layers_query[j](output_raw)
                tmp_key=self.proj_layers_key[j](output_sum)
                tmp_value=self.proj_layers_value[j](output_sum)
                if use_divide_dk:
                    tmp_rst=torch.bmm(nn.Softmax(dim=-1)(torch.bmm(tmp_query, tmp_key.transpose(-2, -1))/((self.hidden_size*2/self.num_heads)**0.5)), tmp_value)
                else:
                    tmp_rst=torch.bmm(nn.Softmax(dim=-1)(torch.bmm(tmp_query, tmp_key.transpose(-2, -1))), tmp_value)
            if use_residual:
                if use_concate_sum:
                    rst.append(torch.cat([tmp_rst+tmp_query, tmp_value], -2))
                else:
                    rst.append(tmp_rst+tmp_query)   ########### +tmp_query
            else:
                if use_concate_sum:
                    rst.append(torch.cat([tmp_rst, tmp_value], -2))
                else:
                    rst.append(tmp_rst)
        rst=torch.cat(rst, -1)
        #if use_residual:
        #     # rst=rst+sum_vec.unsqueeze(-2)
        #    rst+=output_raw
        if use_concate_raw and use_concate_sum:
            raise KeyError('cannot concatenate output_raw and output_sum at the same time.')
        if use_concate_raw:
            rst=torch.cat([output_raw, rst], 2)
        # elif use_concate_sum:
        #     rst=torch.cat([output_sum, rst], 2)
        return rst



class BiLSTM_centric_model(nn.Module):
    '''
        BiLSTM review-centric model
    '''
    def __init__(self, 
                in_size, 
                hid_size, 
                out_classes, 
                num_heads=2, 
                num_layers=2, 
                dropout_rate=0.15, 
                use_residual=False, 
                use_concate_raw=False, 
                use_concate_sum=False, 
                use_layer_norm=False, 
                use_divide_dk=False):
        super(BiLSTM_centric_model, self).__init__()
        self.in_size=in_size
        self.hid_size=hid_size
        self.out_classes=out_classes
        self.num_layers=num_layers
        self.dropout_rate=dropout_rate
        self.dropout_layer=nn.Dropout(p=self.dropout_rate)
        self.use_residual=use_residual
        self.use_concate_raw=use_concate_raw
        self.use_concate_sum=use_concate_sum
        self.use_layer_norm=use_layer_norm
        self.num_heads=num_heads
        self.use_divide_dk=use_divide_dk
        self.lstm_sum=nn.LSTM(self.in_size, self.hid_size, batch_first=True, bidirectional=True)
        self.attention_layers=nn.ModuleList([BiLSTM_centric_layer(self.in_size, self.hid_size, dropout_rate=self.dropout_rate, num_heads=self.num_heads) for i in range(self.num_layers)])
        if self.use_concate_raw:
            self.proj_layers=nn.ModuleList([nn.Linear(self.hid_size*4, self.in_size) for i in range(self.num_layers-1)])
            self.norm_layers=nn.ModuleList([LayerNorm(self.hid_size*4) for i in range(self.num_layers)])
            self.classifier=nn.Linear(self.hid_size*4, self.out_classes)
        elif self.use_concate_sum:
            self.proj_layers=nn.ModuleList([nn.Linear(self.hid_size*2, self.in_size) for i in range(self.num_layers-1)])
            self.norm_layers=nn.ModuleList([LayerNorm(self.hid_size*2) for i in range(self.num_layers)])
            self.classifier=nn.Linear(self.hid_size*2, self.out_classes)
        else:
            self.proj_layers=nn.ModuleList([nn.Linear(self.hid_size*2, self.in_size) for i in range(self.num_layers-1)])
            self.norm_layers=nn.ModuleList([LayerNorm(self.hid_size*2) for i in range(self.num_layers)])
            self.classifier=nn.Linear(self.hid_size*2, self.out_classes)
    def forward(self, in_vec_raw, in_vec_sum,  len_raw, len_sum,  use_norm=False,  use_sum_avg_pooling=True):
        output_sum, (h_n, c_n)=self.lstm_sum(in_vec_sum)
        output_sum=self.dropout_layer(output_sum)
        for i in range(self.num_layers):
            in_vec_raw=self.dropout_layer(self.attention_layers[i](in_vec_raw, 
                                                            in_vec_sum, 
                                                            len_raw, 
                                                            len_sum, 
                                                            use_sum_lstm=False, 
                                                            output_sum=output_sum, 
                                                            use_residual=self.use_residual, 
                                                            use_sum_avg_pooling=use_sum_avg_pooling, 
                                                            use_concate_raw=self.use_concate_raw, 
                                                            use_concate_sum=self.use_concate_sum, 
                                                            use_divide_dk=self.use_divide_dk))
            if self.use_layer_norm:
                in_vec_raw=self.dropout_layer(self.norm_layers[i](in_vec_raw))
            if i!=self.num_layers-1:
                in_vec_raw=self.dropout_layer(self.proj_layers[i](in_vec_raw))
        rst=self.dropout_layer(self.classifier(mean_pooling(in_vec_raw, len_raw)))
        if use_norm:
            rst=nn.Softmax(dim=-1)(rst)
        return rst