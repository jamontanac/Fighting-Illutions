"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.13
"""
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import Train_model


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline([
        node(
            func=Train_model,
            inputs=["params:model_options"],
            outputs=["model",
                     "report"],
            name = "Training_of_model"
            ),
        ])
    
    Resnet_pipeline = pipeline(
        pipe = pipeline_instance,
        parameters={"params:model_options":"params:training_parameters_Resnet"},
        outputs={"model":"Resnet_model",
                "report": "Resnet_report_train_test"},
        namespace="Resnet_pipeline"
    )
        
    RegnetX_pipeline = pipeline(
        pipe = pipeline_instance,
        parameters={"params:model_options":"params:training_parameters_RegnetX"},
        outputs={"model":"Regnet_x_model",
                "report": "Regnet_x_report_train_test"},
        namespace="Regnet_x_pipeline"
    )
    RegnetY_pipeline = pipeline(
        pipe = pipeline_instance,
        parameters={"params:model_options":"params:training_parameters_RegnetY"},
        outputs={"model":"Regnet_y_model",
                "report": "Regnet_y_report_train_test"},
        namespace="Regnet_y_pipeline"
    )
    return Resnet_pipeline+RegnetX_pipeline+RegnetY_pipeline
