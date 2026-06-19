from transformers import SegformerForSemanticSegmentation

def get_segformer_model(checkpoint: str = "nvidia/mit-b3", num_classes: int = 32):
    """
    Instancie et configure le modèle SegFormer pour la segmentation sémantique.
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint, 
        num_labels=num_classes, 
        ignore_mismatched_sizes=True
    )
    return model