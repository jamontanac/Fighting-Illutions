"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline,pipeline

# from Fighting_Illutions.pipelines import  generate_adversarial_examples
# from Fighting_Illutions.pipelines import  training

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    # training_pipe = training.create_pipeline()
    # Adversarial_generation_pipe = generate_adversarial_examples.create_pipline(attack_types= ["FSGM"] )

    # pipelines["Training"] = pipeline(pipelines["training"])
    pipelines["__default__"] = sum(pipelines.values())
    # pipelines["__default__"] = pipelines["generate_adversarial_examples"]
    return pipelines
    # return {
        # "__default__": Adversarial_generation_pipe,
        # "Training": training_pipe,
        # "Adversarial_Generation":Adversarial_generation_pipe
    # }
