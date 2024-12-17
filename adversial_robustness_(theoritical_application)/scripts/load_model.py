import torch
import timm

def load_pretrained(model_name , custom_weights_path , state_dict_path):
    ''' Returns pre trained model based on the model name provides

    args: model name , path to weights , saved model dict
    returns: Model object with custom weights and bias loaded
    '''

    import os

    model_weights = os.listdir(custom_weights_path)

    if timm.is_model_pretrained(model_name):
        pre_trained = timm.create_model(model_name,pretrained=True)

        #changing image shapes for our custom images
        pre_trained.conv1 = torch.nn.Conv2d(3,64,kernel_size=(3,3),padding=(1,1),stride=(1,1),bias=False)

        #Mapping better weights for the adverseral images
        pre_trained.fc.weight = torch.nn.Parameter(model_weights['fc.weight'])

        pre_trained.fc.bias = torch.nn.Parameter(model_weights['fc.bias'])

        pre_trained.load_state_dict(torch.load(state_dict_path))

        return pre_trained
    else:
        raise ValueError("Cannot find the model specified. Please kindly recheck the model value")
    
