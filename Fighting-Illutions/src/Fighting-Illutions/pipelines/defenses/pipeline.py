"""
This is a boilerplate pipeline 'defenses'
generated using Kedro 0.18.13
"""

# from kedro.pipeline import Pipeline, pipeline
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import resize_pad, image_distortion
def defense_generation_templete() -> Pipeline:
    """
    This pipeline will run a specific defense over the adersarial
    It runs the defenses that are being created and then creates a report
    of how well the defenses worked
    """
    return pipeline(
        [
            node(func=resize_pad,
                 inputs=["Adversarial_Data","params:defence_options"],
                 outputs="Resize_Pad_Image",
                 name="Resize_Pad_Defense"
            ),
            # node(func=image_distortion,
            #      inputs=["Adversarial_Data","params:defence_options"],
            #      outputs="Distortion_Image",
            #      name="Distortion_Defense"
            # ),
        ]
    )
def create_pipeline(**kwargs) -> Pipeline:
    
    Attacks = ["DeepFool", "CarliniL2", "FSGM", "PGD"]
    Models = ["Resnet_model","Regnet_x_model","Regnet_y_model"]
    for index, model_ref in enumerate(Models):
        Defense_pipeline = [
            pipeline(pipe=defense_generation_templete(),
                     )
        ]
        if index==0:
            final_pipelines = sum(Defense_pipeline)
        else:
            final_pipelines += sum(Defense_pipeline)

    # return final_pipelines
    return Pipeline([])

