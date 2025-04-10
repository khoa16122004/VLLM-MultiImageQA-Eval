def init_dataset(dataset_name):
    if dataset_name == "MRAG":
        from dataset import MRAG
        dataset = MRAG()
    
    return dataset


def init_lvlm_model(pretrained, model_name=None):
    
    """
    
        Initialize lvlm model for multiple image input VQA
    
        Args: 
            pretrained: pretrained VLLM model name
            model_name: model name
        
        Return:
            model: lvlm model
            image_token: image placholder for each model
            special_token: special token for each model
                
    """
    
    
    special_token = None
    if "llava" in model_name:
        from models.llava_ import LLava
        image_token = "<image>"
        model = LLava(pretrained, model_name)

    elif "openflamingo" in model_name:
        from models.openflamingo_ import OpenFlamingo
        image_token = "<image>"
        special_token = "<|endofchunk|>"

        model = OpenFlamingo(pretrained)
    
    elif "mantis" in model_name:
        from models.mantis_ import Mantis
        image_token = "<image>"
        model = Mantis(pretrained)
        
    elif "deepseek" in model_name:
        from models.deepseek_ import DeepSeek
        image_token = "<image_placeholder>"
        model = DeepSeek(pretrained)
    
    return model, image_token, special_token


def init_encode_model(model_name):
    
    """
        initialize encode model for multiple image input VQA
    """
    
    if model_name == "ReT":
        from encode_model import ReTWrapper
        model =  ReTWrapper()
        dim = 4096
    elif model_name == "CLIP":
        from encode_model import MyCLIPWrapper
        model =  MyCLIPWrapper()
        dim = 512
    return model, dim   

        
