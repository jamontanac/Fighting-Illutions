"""
This is a boilerplate pipeline 'generate_adversarial_examples'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import classification, Fast_gradient_attack


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance=pipeline([
        node(
            func=classification,
            inputs=["model"],
            outputs="Classifier",
            name="ART_classifier"
        ),
        node(func=Fast_gradient_attack,
             inputs=["Classifier","params:attack"],
             outputs="AdversarialDataFSGM",
             name="FSGM")
    ],inputs=["model"],outputs=["AdversarialDataFSGM"])
    
    Resnet_pipeline = pipeline(
        pipe=pipeline_instance, 
        inputs={"model":"Resnet_model"}, 
        outputs={"AdversarialDataFSGM":"Resnet_FSGM"}, 
        parameters={"params:attack": "params:Fast_Signed_attack"}, 
        namespace="Resnet_attack"
        )
    RegnetX_pipeline = pipeline(
        pipe=pipeline_instance, 
        inputs={"model":"Regnet_x_model"}, 
        outputs={"AdversarialDataFSGM":"Regnet_x_FSGM"}, 
        parameters={"params:attack": "params:Fast_Signed_attack"}, 
        namespace="Regnet_x_attack"
        )
    
    RegnetY_pipeline = pipeline(
        pipe=pipeline_instance, 
        inputs={"model":"Regnet_y_model"}, 
        outputs={"AdversarialDataFSGM":"Regnet_y_FSGM"}, 
        parameters={"params:attack": "params:Fast_Signed_attack"}, 
        namespace="Regnet_y_attack"
        )
    
    return Resnet_pipeline + RegnetX_pipeline+ RegnetY_pipeline
