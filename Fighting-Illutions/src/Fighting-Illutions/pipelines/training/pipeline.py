"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.13
"""
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import Train_model,plot_results


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline([
        node(
            func=Train_model,
            inputs=["params:model_options"],
            outputs=["model",
                     "report"],
            name = "Training_of_model"
            ),
        ],outputs=["model","report"])
    plot_pipeline = pipeline([
            node(
                func=plot_results,
                inputs="report",
                outputs="figure"
                ),
    ])

    Resnet_pipeline = pipeline(
        pipe = pipeline_instance,
        parameters={"params:model_options":"params:training_parameters_Resnet"},
        outputs={"model":"Resnet_model",
                "report": "Resnet#_report_train_test"},
        namespace="Resnet_pipeline"
    ) + pipeline(pipe=plot_pipeline,
                 inputs={"report": "Resnet#_report_train_test"},
                 outputs={"figure": "Resnet#_plot_result"},
                 namespace="Resnet_report_plot")
        
    RegnetX_pipeline = pipeline(
        pipe = pipeline_instance,
        parameters={"params:model_options":"params:training_parameters_RegnetX"},
        outputs={"model":"Regnet_x_model",
                "report": "Regnet_x#_report_train_test"},
        namespace="Regnet_x_pipeline"
    )+  pipeline(pipe=plot_pipeline,
                 inputs={"report": "Regnet_x#_report_train_test"},
                 outputs={"figure": "Regnet_x#_plot_result"},
                 namespace="Regnet_x_report_plot")

    RegnetY_pipeline = pipeline(
        pipe = pipeline_instance,
        parameters={"params:model_options":"params:training_parameters_RegnetY"},
        outputs={"model":"Regnet_y_model",
                "report": "Regnet_y#_report_train_test"},
        namespace="Regnet_y_pipeline"
    )+  pipeline(pipe=plot_pipeline,
                 inputs={"report": "Regnet_y#_report_train_test"},
                 outputs={"figure": "Regnet_y#_plot_result"},
                 namespace="Regnet_y_report_plot")

    return Resnet_pipeline+RegnetX_pipeline+RegnetY_pipeline
