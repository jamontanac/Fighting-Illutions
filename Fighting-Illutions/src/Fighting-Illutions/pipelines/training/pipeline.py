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
                     "loss_train",
                     "accuracy_train",
                     "loss_test",
                     "accuracy_test"],
            name = "Training_of_model"
            ),
        ])
    
    Resnet_pipeline = pipeline(
        pipe = pipeline_instance,
        parameters={"params:model_options":"params:training_parameters_Resnet"},
        outputs={"model":"Resnet_model",
                "loss_train":"Resnet_loss_train",
                "accuracy_train":"Resnet_accuracy_train",
                "loss_test":"Resnet_loss_test",
                "accuracy_test": "Resnet_accuracy_test"},
        namespace="Resnet_pipeline"
    )
        
    RegnetX_pipeline = pipeline(
        pipe = pipeline_instance,
        parameters={"params:model_options":"params:training_parameters_RegnetX"},
        outputs={"model":"Regnet_x_model",
                "loss_train":"Regnet_x_loss_train",
                "accuracy_train":"Regnet_x_accuracy_train",
                "loss_test":"Regnet_x_loss_test",
                "accuracy_test": "Regnet_x_accuracy_test"},
        namespace="Regnet_x_pipeline"
    )
    RegnetY_pipeline = pipeline(
        pipe = pipeline_instance,
        parameters={"params:model_options":"params:training_parameters_RegnetY"},
        outputs={"model":"Regnet_y_model",
                "loss_train":"Regnet_y_loss_train",
                "accuracy_train":"Regnet_y_accuracy_train",
                "loss_test":"Regnet_y_loss_test",
                "accuracy_test": "Regnet_y_accuracy_test"},
        namespace="Regnet_y_pipeline"
    )
    return Resnet_pipeline+RegnetX_pipeline+RegnetY_pipeline
