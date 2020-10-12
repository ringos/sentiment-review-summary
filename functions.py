import torch
import torch.nn as nn

batch_size=128

def test(model,test_loader,use_sum,use_sum_avg_pooling):
    correct=0
    total_num=0
    criterion=nn.CrossEntropyLoss().to('cuda')
    total_loss=0.0
    for i,(test_raw,test_label,test_sum) in enumerate(test_loader):
        model.eval()
        test_raw_ids=test_raw[0].to('cuda')
        test_raw_len=test_raw[1].to('cuda')
        if use_sum:
            test_sum_ids=test_sum[0].to('cuda')
            test_sum_len=test_sum[1].to('cuda')
        test_label=test_label.to('cuda').to(dtype=torch.long)
        
        if not use_sum:
            prediction_tensor=model(test_raw_ids,test_raw_len,use_norm=False)
        else:
            prediction_tensor=model(test_raw_ids,test_sum_ids,test_raw_len,test_sum_len,use_norm=False,use_sum_avg_pooling=use_sum_avg_pooling)
        with torch.no_grad():
            loss=criterion(prediction_tensor,test_label)
            total_loss+=loss.item()
#        print(prediction_tensor.size())
        prediction_tensor=nn.Softmax(dim=-1)(prediction_tensor)
        predictions=prediction_tensor.argmax(1)
        for j,sample in enumerate(predictions):
            total_num+=1
            if predictions[j]==test_label[j]:
                correct+=1
    rst=dict()
    accuracy=correct/total_num
    rst['accuracy']=accuracy
    rst['loss']=total_loss/len(test_loader)/batch_size
    return rst

