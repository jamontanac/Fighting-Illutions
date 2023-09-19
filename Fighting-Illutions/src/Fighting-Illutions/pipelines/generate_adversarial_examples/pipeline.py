"""
This is a boilerplate pipeline 'generate_adversarial_examples'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import classification, Adversarial_generation
from typing import List
from kedro.config import ConfigLoader
from kedro.framework.project import settings

# conf_loader = OmegaConfigLoader(conf_source=settings.CONF_SOURCE)
# print(conf_loader.config_patterns.get("parameters"))
conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
# print(conf_loader.get("parameters*"))
parameters = conf_loader["parameters"]
# print(parameters["Attacks_to_use"]["attacks"])

def new_attack_generation_template() -> Pipeline:
    """
    This 2 node pipeline will generate the classifier 
    for a pytorch model and generate the adversarial examples
    for a specific attack
    returns:
        Pipeline: this specific pipeline has been designed in a way
        that allows the user to implement different parametrised 
        art attacks for exploring further combinations of parameters
        via a modular pipeline instance.
    """
    return pipeline(
        [
            node(
                func=classification,
                inputs=["model"],
                outputs="Classifier",
                name="ART_Classifier"
            ),
            # node(
            #     func=Evasion_Attack,
            #     inputs=["Classifier","params:attack_options"],
            #     outputs="Adversarial_Attack",
            #     name="ART_adversarial_attack"
            # ),
            node(
                func=Adversarial_generation,
                inputs=["Classifier","params:attack_options"],
                outputs="Adversarial_Data",
                name="Generation_Adversarial_Data"
            )
        ],
        inputs=["model"],
        outputs=["Adversarial_Data"]
    )


def create_pipeline(attack_types:List[str]=parameters["Attacks_to_use"]["attacks"]) -> Pipeline:
    """This function will create a complete modelling
    pipeline that consolidates a single shared 'model' stage,
    several modular instances of the 'training' stage
    and returns a single, appropriately namespaced Kedro pipeline
    object:
    ┌───────────────────────────────┐
    │                               │
    │        ┌────────────┐         │
    │     ┌──┤    Model   ├───┐     │
    │     │  └──────┬─────┘   │     │
    │     │         │         │     │
    │ ┌───┴───┐ ┌───┴───┐ ┌───┴───┐ │
    │ │Attack │ │Attack │ │Attack │ │
    │ │ Type  │ │ Type  │ │  Type │ │
    │ │   1   │ │   2   │ │   n.. │ │
    │ └───────┘ └───────┘ └───────┘ │
    │                               │
    └───────────────────────────────┘

    Args:
        attack_types (List[str]): The instances of ART attacks
            we want to build, each of these must correspond to
            parameter keys of the same name

    Returns:
        Pipeline: A single pipeline encapsulating the training
            stage as well as the adversarial generation of the samples.
    """
    for index, model_ref in enumerate(["Resnet_model","Regnet_x_model","Regnet_y_model"]):
        attack_pipelines = [
            pipeline(pipe=new_attack_generation_template(),
                    parameters={"params:attack_options":f"params:Adversarial_Attacks.{attack_type}"},
                    inputs={"model":f"{model_ref}"},
                    outputs={"Adversarial_Data":f"{model_ref}_Adversarial_{attack_type}@Dataset"},
                    namespace=f"{model_ref}_Adversarial_Generation_{attack_type}")
                    for attack_type in attack_types
                    ]
        if index ==0:
            final_pipelines = sum(attack_pipelines)
        else:
            final_pipelines+= sum(attack_pipelines)

    
    # attacks_pipeline= pipeline(
    #     pipe=final_pipelines,
    #     namespace="Adversrial_Attacks"
    # )
    
    return final_pipelines

# def create_pipeline(**kwargs) -> Pipeline:
#     pipeline_instance=pipeline([
#         node(
#             func=classification,
#             inputs=["model"],
#             outputs="Classifier",
#             name="ART_classifier"
#         ),
#         node(func=Fast_gradient_attack,
#              inputs=["Classifier","params:attack"],
#              outputs="AdversarialDataFSGM",
#              name="FSGM")
#     ],inputs=["model"],outputs=["AdversarialDataFSGM"])
    
#     Resnet_pipeline = pipeline(
#         pipe=pipeline_instance, 
#         inputs={"model":"Resnet_model"}, 
#         outputs={"AdversarialDataFSGM":"Resnet_FSGM"}, 
#         parameters={"params:attack": "params:Fast_Signed_attack"}, 
#         namespace="Resnet_attack"
#         )
#     RegnetX_pipeline = pipeline(
#         pipe=pipeline_instance, 
#         inputs={"model":"Regnet_x_model"}, 
#         outputs={"AdversarialDataFSGM":"Regnet_x_FSGM"}, 
#         parameters={"params:attack": "params:Fast_Signed_attack"}, 
#         namespace="Regnet_x_attack"
#         )
    
#     RegnetY_pipeline = pipeline(
#         pipe=pipeline_instance, 
#         inputs={"model":"Regnet_y_model"}, 
#         outputs={"AdversarialDataFSGM":"Regnet_y_FSGM"}, 
#         parameters={"params:attack": "params:Fast_Signed_attack"}, 
#         namespace="Regnet_y_attack"
#         )
    
#     return Resnet_pipeline + RegnetX_pipeline+ RegnetY_pipeline
