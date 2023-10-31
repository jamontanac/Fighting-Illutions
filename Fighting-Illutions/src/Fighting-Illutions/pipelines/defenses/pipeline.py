"""
This is a boilerplate pipeline 'defenses'
generated using Kedro 0.18.13
"""

# from kedro.pipeline import Pipeline, pipeline
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import Padding_defense, Distort_defense, Padding_Distort_defense, Report 
def defense_generation_templete() -> Pipeline:
    """
    This pipeline will run a specific defense over the adersarial
    It runs the defenses that are being created and then creates a report
    of how well the defenses worked
    """
    return pipeline(
        [
            node(func=Padding_defense,
                 inputs=["Adversarial_Data","params:defense_options"],
                 outputs="Padding_Defense_Loader_Image",
                 name="Padding_Defense"
            ),
            node(func=Distort_defense,
                 inputs=["Adversarial_Data","params:defense_options"],
                 outputs="Distortion_Defense_Loader_Image",
                 name="Distortion_Defense"
            ),
            node(func=Padding_Distort_defense,
                 inputs=["Adversarial_Data","params:defense_options"],
                 outputs="Padding_Distortion_Defense_Loader_Image",
                 name="Padding_Distortion_Defense"
                ),
            node(func=Report,
                 inputs=["Padding_Defense_Loader_Image","Model","params:Report_params"],
                 outputs=["Accuracy_report_Padding",
                          "KL_report_Padding",
                          "Conf_matrix_Padding",
                          "Distributions_Padding"],
                 name="Report_Padding"),
            node(func=Report,
                 inputs=["Distortion_Defense_Loader_Image","Model","params:Report_params"],
                 outputs=["Accuracy_report_Distortion",
                          "KL_report_Distortion",
                          "Conf_matrix_Distortion",
                          "Distributions_Distortion"],
                 name="Report_Distortion"),
            node(func=Report,
                 inputs= ["Padding_Distortion_Defense_Loader_Image","Model","params:Report_params"],
                 outputs= ["Accuracy_report_Padding_Distortion",
                           "KL_report_Padding_Distortion",
                           "Conf_matrix_Padding_Distortion",
                           "Distributions_Padding_Distortion"],
                 name="Report_Padding_Distortion")
        ],inputs = ["Adversarial_Data","Model"],
        outputs = ["Accuracy_report_Padding",
                   "KL_report_Padding",
                   "Conf_matrix_Padding",
                   "Distributions_Padding",
                   "Accuracy_report_Distortion",
                   "KL_report_Distortion",
                   "Conf_matrix_Distortion",
                   "Distributions_Distortion",
                   "Accuracy_report_Padding_Distortion",
                   "KL_report_Padding_Distortion",
                   "Conf_matrix_Padding_Distortion",
                   "Distributions_Padding_Distortion",
                   ]

    )
def create_pipeline(**kwargs) -> Pipeline:
    
    Attacks = ["DeepFool", "CarliniL2", "FSGM", "PGD"]
    Models = ["Resnet_model","Regnet_x_model","Regnet_y_model"]
    Defenses = ["Padding","Distortion","Padding_Distortion"]
    Defense_pipeline = []
    for index, model_ref in enumerate(Models):
        for attack_type in Attacks:
            output_result = {f"Accuracy_report_{defense}": f"{model_ref}_Accuracy_{defense}_{attack_type}" for defense in Defenses}
            output_result.update({f"KL_report_{defense}":f"{model_ref}_KLdiff_{defense}_{attack_type}" for defense in Defenses})
            output_result.update({f"Conf_matrix_{defense}":f"{model_ref}_confusionMatrix_{defense}_{attack_type}" for defense in Defenses})
            output_result.update({f"Distributions_{defense}":f"{model_ref}_Distribution_{defense}_{attack_type}" for defense in Defenses})
            
            tmp_pipeline = pipeline( pipe=defense_generation_templete(),
                     parameters={"params:defense_options":f"params:Parameters_defenses.{model_ref}",
                                 "params:Report_params":f"params:Parameters_defenses"},
                     inputs={"Adversarial_Data":f"{model_ref}_Adversarial_{attack_type}@Dataset","Model":model_ref},
                     outputs=output_result,
                     namespace=f"{model_ref}_Defenses_{attack_type}")
            Defense_pipeline.append(tmp_pipeline)
        if index == 0:
            final_pipelines = sum(Defense_pipeline)
        else:
            final_pipelines += sum(Defense_pipeline)

    return final_pipelines

